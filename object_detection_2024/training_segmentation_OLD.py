"""
File containing the TrainingSegmentation class, the main class for training and validating the segmentation model.
"""

import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from time import time
from datetime import datetime as dt
from datetime import timedelta
from tqdm.notebook import tqdm, trange  # Just to log fancy training progress bar

from utils_segmentation.utils import set_seed, collate_fn, compute_iou_mask, compute_iou_box
from dataset_segmentation import LFASegmentationDataset
from model_segmentation import get_segmentation_model

class TrainingSegmentation:
    def __init__(self, config_file, kit_id, transformation_train=None, transformation_val=None):

        self.kit_id = kit_id

        # Configuration file
        self.config = config_file
        self.data_settings = self.config['DataSettings']
        self.parameters = self.config['TrainingParameters']
        self.data_dir = self.data_settings['data_dir']

        # Separate into classes (kit and membrane)
        self.classes = self.data_settings['classes']
        self.class_ids = self.data_settings['class_ids']
        self.n_classes = len(self.classes)
        
        # Hyperparameters
        self.batch_size = self.parameters['batch_size']
        self.num_workers = self.parameters['num_workers']
        self.epochs = self.parameters['num_epochs']
        self.save_path = self.parameters['save_path']  #  To save model
        
        # Seed
        if self.parameters['seed'] is None:
            self.parameters['seed'] = int(dt.now().timestamp())
            
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.parameters['device'] = self.device
        print(f'Using {self.device} device')

        # Metrics
        self.metrics_train = {'loss_classifier': [], 
                                   'loss_box_reg': [], 
                                   'loss_mask': [], 
                                   'loss_objectness': [], 
                                   'loss_rpn_box_reg': [],
                                   'total_loss': []}

        init_zeros = np.zeros([1, self.n_classes])
        self.metrics_val = {'mask': {
                              'precision': init_zeros.copy(), 
                              'recall': init_zeros.copy(),
                              'f1': init_zeros.copy(),
                              'iou_th': init_zeros.copy(),
                              'iou_avg': init_zeros.copy()},
                            'box': {
                              'precision': init_zeros.copy(), 
                              'recall': init_zeros.copy(),
                              'f1': init_zeros.copy(),
                              'iou_th': init_zeros.copy(),
                              'iou_avg': init_zeros.copy()}}

        # Transformations
        self.transformation_train = transformation_train
        self.transformation_val = transformation_val

        # Model design
        set_seed(self.parameters['seed'])
        self.model = get_segmentation_model(num_classes=self.parameters['num_classes'], 
                                        hidden_size=self.parameters['hidden_size']).to(self.device)
        
        self.optimizer = Adam(params=self.model.parameters(), lr=float(self.parameters['learning_rate']))
        
        # Split filenames into training and validation 
        self.train_val_ratio = self.parameters['train_validation_ratio']
        self.image_path = os.path.join(self.data_dir, f'{self.kit_id}_train_images')
        self.filenames_train, self.filenames_val = self.split_train_val_filenames()
        
        # Load training and validation datasets and dataloaders
        self.loader_train, self.loader_val = self.init_dataloader()
        self.n_train = len(self.loader_train)
        self.n_val = len(self.loader_val)

        # Time stamp
        self.stamp = None

    def main(self, save_bool=False):
        """
        Here is where the magic happens. It trains and validates the model for all epochs, store the training and
        validation metrics, display the results and save the best models.
        """
        
        # To save the model
        best_iou_mean = 0.0
        time_str = dt.now().strftime('%Y-%m-%d_%H.%M.%S')  # Timestamp to identify training runs
        self.stamp = time_str
        file_path = f'{self.save_path}/{time_str}'

        # To log fancy progress bar
        epoch_loop = tqdm(range(1, self.epochs + 1), total=self.epochs)
        epoch_loop.set_description(f"Training for {self.epochs} epochs")  # Description
        
        start = time()  #  For elapsed time
        for epoch_ndx in epoch_loop:

            # Train and return epoch train loss
            epoch_train_loss = self.train(epoch_ndx, self.loader_train, self.n_train)
            
            # Update train metrics
            for j, k in enumerate(self.metrics_train.keys()):
                self.metrics_train[k].append(epoch_train_loss[j])

            # Validate, update metrics and return F1 score and average IoU
            iou_mask, iou_box = self.validate(epoch_ndx, self.loader_val, self.n_val)
            iou_mean  = np.append(iou_mask, iou_box).mean()

            # Update progress bar
            epoch_loop.set_postfix_str(f"Train loss = {epoch_train_loss[-1]:.3f}, IoU mask, box, mean = {np.round(iou_mask, 3)}, {np.round(iou_box, 3)}, {np.round(iou_mean, 3)}")

            if save_bool:
                end = time()
                elapsed_time = str(timedelta(seconds=int(end-start)))
                if iou_mean > best_iou_mean:
                    best_iou_mean = iou_mean
                    self.save_model(epoch_ndx, file_path, stamp='_best', elapsed_time=elapsed_time)
                else:
                    self.save_model(epoch_ndx, file_path, stamp='', elapsed_time=elapsed_time)

          

    def init_dataloader(self):
        """
        Initialize train and validation dataloaders by splitting the original train dataset
        """
        
        print("Loading data...")
        # Train and validation datasets
        dataset_train = LFASegmentationDataset(self.data_settings, 
                                               kit_id=self.kit_id,
                                               dataset='train',
                                               filenames=self.filenames_train,
                                               transforms=self.transformation_train)

        dataset_val = LFASegmentationDataset(self.data_settings, 
                                              kit_id=self.kit_id,
                                              dataset='train',
                                              filenames=self.filenames_val,
                                              transforms= self.transformation_val)
        # Initialize data loaders
        loader_train = DataLoader(dataset=dataset_train,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=self.num_workers,
                                  collate_fn=collate_fn,
                                  pin_memory=True)

        loader_val = DataLoader(dataset=dataset_val,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 num_workers=self.num_workers,
                                 collate_fn=collate_fn,
                                 pin_memory=True)
        
        return loader_train, loader_val

    def split_train_val_filenames(self):
        """
        Split filenames into training and validation subsets
        """
        
        filenames = [path.replace('.jpg', '') for path in os.listdir(self.image_path)]
        
        random.shuffle(filenames)
        
        n_train_old = len(filenames)
        n_train_new = int(self.train_val_ratio * n_train_old)
        
        filenames_train, filenames_val = filenames[:n_train_new], filenames[n_train_new: ]
        assert len(filenames_train) + len(filenames_val) == n_train_old, 'Wrong dataset lenghts!'
        
        return filenames_train, filenames_val
        
    def train(self, epoch_ndx, trainloader, n_train):
        """
        Train model for one epoch and return training total loss
        """

        # To store running train metrics (loss)
        running_loss = np.zeros(len(self.metrics_train))
    
        # To log fancy progress bar
        train_loop = tqdm(enumerate(trainloader, start=1), total=n_train)
        train_loop.set_description(f"Epoch {epoch_ndx} | Training")
        
        self.model.train()
        for i, (images, targets) in train_loop:

            # Send to device
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            # Set gradients to zero
            self.optimizer.zero_grad()

            # Forward pass and loss calculation (all in one)
            loss_dict = self.model(images, targets)
            total_loss = sum(loss for loss in loss_dict.values())
            
            # Backward pass
            total_loss.backward()
            
            # Optimize
            self.optimizer.step()

            # Update running loss
            running_loss += [v.item() for k, v in loss_dict.items()] + [total_loss.item()]

            # Update progress bar
            train_loop.set_postfix_str(f"Batch loss = {total_loss.item():.3f}")
        
        return running_loss/n_train
    
    @torch.no_grad()
    def validate(self, epoch_ndx, loader, n_loader):
        """
        Validate the model over the entire validation set and calculate several performance metrics.
        """
        
        # Set model to evaluation mode
        self.model.eval()
        
        # List of thresholds to compute average precision curve and identify optimal thresholds for latter inference
        iou_thresholds = np.arange(0.5, 0.95, 0.05)
        n_thresholds = len(iou_thresholds)

        # Initialize metrics for masks and boxes for all classes
        zeros_init = np.zeros((self.n_classes, n_thresholds), dtype=int)
        metrics = {'mask': {'tp': zeros_init.copy(), 
                            'fp': zeros_init.copy(), 
                            'fn': zeros_init.copy(),
                            'iou': np.zeros([1, self.n_classes])},  #  Average IoU 
                   'box':  {'tp': zeros_init.copy(), 
                            'fp': zeros_init.copy(), 
                            'fn': zeros_init.copy(),
                            'iou': np.zeros([1, self.n_classes])}}

        # To log fancy progress bar
        val_loop = tqdm(loader, total=n_loader)
        val_loop.set_description(f"Epoch {epoch_ndx} | Validation")
    
        for images, targets in val_loop:    

            # Send to device and to list
            images = list(img.to(self.device) for img in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            # Inference step
            predictions = self.model(images)
            
            # Keep track of average IoU values over batch
            batch_iou_mask = np.array([0.0, 0.0])
            batch_iou_box = np.array([0.0, 0.0])
            
            for img, target, pred in zip(images, targets, predictions):

                # Get labels and scores, which are aligned
                labels = pred['labels'].tolist()
                scores = pred['scores'].tolist()
                assert len(labels) == len(scores)

                # Get boxes and masks, which are also naturally aligned
                boxes = pred['boxes']
                masks = pred['masks']
                assert boxes.shape[0] == masks.shape[0] == len(labels)

                # Loop over classes (i.e. kit and membrane)
                for i, cls in enumerate(self.classes):
                    if self.class_ids[i] in labels:  # Check whether there is at least one prediction for that class

                        # Get the maximum confidence class location (i.e. first occurrence in list)
                        class_loc = labels.index(self.class_ids[i])

                        # Get best class score 
                        class_score = scores[class_loc]

                        # Get best class boxes and masks
                        class_box = boxes[class_loc]
                        class_mask = masks[class_loc, 0]

                        # Binarize masks
                        class_mask = (class_mask >= self.parameters['mask_thresholds'][i]).to(torch.uint8)

                        # Compute IoU
                        class_iou_mask = compute_iou_mask(class_mask, target['masks'][i])
                        class_iou_box = compute_iou_box(class_box, target['boxes'][i])
                        # Update batch IoU
                        batch_iou_mask[i] += class_iou_mask
                        batch_iou_box[i] += class_iou_box
                        
                        # Calculate TP and FP for each IoU threshold and update metrics (mask)
                        class_tp_mask = np.array([class_iou_mask >= iou_th for iou_th in iou_thresholds])  # List of booleans
                        metrics['mask']['tp'][i] += class_tp_mask
                        metrics['mask']['fp'][i] += ~(class_tp_mask)  # Switch booleans

                        # Calculate TP and FP for each IoU threshold and update metrics (bbox)
                        class_tp_box = np.array([class_iou_box >= iou_th for iou_th in iou_thresholds])  # List of booleans
                        metrics['box']['tp'][i] += class_tp_box
                        metrics['box']['fp'][i] += ~(class_tp_box)  # Switch booleans

                    else:  # If there is no prediction, we consider the classification outcome as FN for all IoU thresholds
                        print(f'{cls} is missing from the prediction!')
                        metrics['mask']['fn'][i] += np.ones_like(iou_thresholds, dtype=int)
                        metrics['box']['fn'][i] += np.ones_like(iou_thresholds, dtype=int)

            # Update IoU averages
            n_images = len(images)
            metrics['mask']['iou'] += batch_iou_mask/n_images
            metrics['box']['iou'] += batch_iou_box/n_images

            # Update progress bar
            val_loop.set_postfix_str(f"IoU Mask = {np.round(batch_iou_mask/n_images, 3)}, IoU Box = {np.round(batch_iou_box/n_images, 3)}")
            
                
        # To regulate precision and recall calculations
        epsilon = 1E-5
        for key in ['mask', 'box']:
            tp, fp, fn = metrics[key]['tp'], metrics[key]['fp'], metrics[key]['fn']
            precision = (tp + epsilon)/ (tp + fp + epsilon)
            recall = (tp + epsilon)/(tp + fn + epsilon)
            f1 = 2 * recall * precision / (recall + precision)

            # Select metrics where f1 is max
            max_loc = (range(0, f1.shape[0]), np.argmax(f1, axis=1))
            precision_max = np.expand_dims(precision[max_loc], 0)
            recall_max = np.expand_dims(recall[max_loc], 0)
            f1_max = np.expand_dims(f1[max_loc], 0)
            threshold_max = np.expand_dims(iou_thresholds[max_loc[1]], 0)

            # Update metrics
            self.metrics_val[key]['precision'] = np.concatenate([self.metrics_val[key]['precision'], precision_max])
            self.metrics_val[key]['recall'] = np.concatenate([self.metrics_val[key]['recall'], recall_max])
            self.metrics_val[key]['f1'] = np.concatenate([self.metrics_val[key]['f1'], f1_max])
            self.metrics_val[key]['iou_th'] = np.concatenate([self.metrics_val[key]['iou_th'], threshold_max])

        # Compute average IoU over validation set
        iou_avg_mask = metrics['mask']['iou']/n_loader
        iou_avg_box = metrics['box']['iou']/n_loader
        self.metrics_val['mask']['iou_avg'] = np.concatenate([self.metrics_val['mask']['iou_avg'], iou_avg_mask])
        self.metrics_val['box']['iou_avg'] = np.concatenate([self.metrics_val['box']['iou_avg'], iou_avg_box])

        return iou_avg_mask.squeeze(), iou_avg_box.squeeze()


    
    def get_metrics(self):
      """
      Returns metrics as Pandas DataFrames ready to be plotted
      """

      metrics_val_reduced = {}
      for metric in ['f1', 'iou_avg']:
        for key in ['mask', 'box']:
            metrics_val_reduced[f'{key}_{metric}_kit'] = self.metrics_val[key][metric][1:, 0]  # Excludes first zero element
            metrics_val_reduced[f'{key}_{metric}_mebrane'] = self.metrics_val[key][metric][1:, 1]
      metrics_val_df = pd.DataFrame.from_dict(metrics_val_reduced)
      metrics_val_df.index.name = 'epochs'
      
      metrics_train_df = pd.DataFrame.from_dict(self.metrics_train)
      metrics_train_df.index.name = 'epochs'

      return metrics_train_df, metrics_val_df

    def save_model(self, epoch_ndx, file_path, stamp, elapsed_time):
        """
        Save model state and hyperparameters
        """
        state = {
            'model_state': self.model.state_dict(),  # model's state
            'hyperparameters': self.parameters,
            'epoch': epoch_ndx,
            'metrics': {'train': self.metrics_train, 'val': self.metrics_val},
            'stamp': self.stamp,
            'elapsed_time': elapsed_time
        }
        torch.save(state, file_path + stamp + '.state')


    def load_state(self, path, device):
        """
        Load model state and metrics
        """
        state = torch.load(path, map_location=torch.device(device))
        self.model.load_state_dict(state['model_state'])
        self.metrics_train = state['metrics']['train']
        self.metrics_val = state['metrics']['val']
        self.stamp = state['stamp']