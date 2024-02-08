"""
File containing the TrainingSegmentation class, the main class for training and validating the segmentation model.
"""

import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler

from time import time
from datetime import datetime as dt
from datetime import timedelta
from tqdm.notebook import tqdm, trange  # Just to log fancy training progress bar

# Custom packages
from utils_segmentation.utils import set_seed, collate_fn, compute_iou_mask, compute_iou_box
from dataset_segmentation import LFASegmentationDataset
from model_segmentation import get_segmentation_model


class TrainingSegmentation:
    def __init__(self, config_file, kit_id, transformation_train=None, transformation_val=None):

        self.kit_id = kit_id

        # Configuration file
        self.config_file = config_file
        self.data_settings = self.config_file['DataSettings']
        self.parameters = self.config_file['TrainingParameters']
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

        self.metrics_val = {'score_kit': [],
                            'score_membrane': [],
                            'iou_mask_kit': [],
                            'iou_mask_membrane': [],
                            'iou_box_kit': [],
                            'iou_box_membrane': []}

        # Transformations
        self.transformation_train = transformation_train
        self.transformation_val = transformation_val

        # Set seed for reproducibility
        set_seed(self.parameters['seed'])

        # Split filenames into training and validation
        self.train_val_ratio = self.parameters['train_validation_ratio']
        self.image_path = os.path.join(self.data_dir, f'{self.kit_id}_train_images')
        self.filenames_train, self.filenames_val = self.split_train_val_filenames()

        # Load training and validation datasets and dataloaders
        self.loader_train, self.loader_val = self.init_dataloader()
        self.n_train = len(self.loader_train)
        self.n_val = len(self.loader_val)

        # Model architecture, optimizer and scheduler
        self.model = get_segmentation_model(num_classes=self.parameters['num_classes'],
                                        hidden_size=self.parameters['hidden_size']).to(self.device)

        self.optimizer = Adam(params=self.model.parameters(), lr=float(self.parameters['learning_rate']))

        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)

        # Time stamp to identify unique instances of the class
        self.stamp = None

    def main(self, save_bool=False):
        """
        Here is where the magic happens. It trains and validates the model for all epochs, store the training and
        validation metrics, display the results and save the best models.
        """

        # To keep track of best model
        best_iou_mean = 0.0  

        # To save the model
        time_str = dt.now().strftime('%Y-%m-%d_%H.%M.%S')  # Timestamp to identify training runs
        self.stamp = time_str
        file_path = f'{self.save_path}/{time_str}'

        # To log fancy progress bar
        epoch_loop = tqdm(range(1, self.epochs + 1), total=self.epochs)
        epoch_loop.set_description(f"Training for {self.epochs} epochs")  # Description

        start = time()  #  For elapsed time
        for epoch_ndx in epoch_loop:

            # Train and return epoch train loss
            epoch_train_loss = self.train(epoch_ndx)

            # Update train metrics
            for j, k in enumerate(self.metrics_train.keys()):
                self.metrics_train[k].append(epoch_train_loss[j])

            # Update scheduler and log the change in learning rate
            before_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()
            after_lr = self.optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch_ndx}: lr {before_lr} -> {after_lr}")

            # Validate, update validation metrics and return average mask and box IoU, for each class
            iou_mask, iou_box = self.validate(epoch_ndx)
            iou_mean = np.append(iou_mask, iou_box).mean()  # Mean over four IoUs

            # Update progress bar
            epoch_loop.set_postfix_str(f"Train loss = {epoch_train_loss[-1]:.3f}, IoU mask, box, mean = {np.round(iou_mask, 3)}, {np.round(iou_box, 3)}, {np.round(iou_mean, 3)}")

            # Save model (in separate file if it is the best)
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

    def train(self, epoch_ndx):
        """
        Train model for one epoch and return training total loss
        """

        # To store running train metrics (loss)
        running_loss = np.zeros(len(self.metrics_train))

        # To log fancy progress bar
        train_loop = tqdm(enumerate(self.loader_train, start=1), total=self.n_train)
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

        return running_loss/self.n_train

    @torch.no_grad()
    def validate(self, epoch_ndx):
        """
        Validate the model over the entire validation set, calculate and update performance metrics (IoU)
        """

        # Set model to evaluation mode
        self.model.eval()
        
        # Final metrics over entire validation set
        scores_avg = np.zeros(self.n_classes)
        iou_masks_avg = np.zeros(self.n_classes)
        iou_boxes_avg = np.zeros(self.n_classes)

        # To log fancy progress bar
        val_loop = tqdm(self.loader_val, total=self.n_val)
        val_loop.set_description(f"Epoch {epoch_ndx} | Validation")
        
        for images, targets in val_loop:
            # Send to device and to list
            images = list(img.to(self.device) for img in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            # Inference step
            predictions = self.model(images)
            
            # Keep track of batch metrics to log
            batch_score = np.zeros(self.n_classes)
            batch_iou_mask = np.zeros(self.n_classes)
            batch_iou_box = np.zeros(self.n_classes)

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

                        # Update dictionary
                        batch_score[i] += class_score
                        batch_iou_mask[i] += class_iou_mask
                        batch_iou_box[i] += class_iou_box

                    else:  # If there is no prediction, we leave all zeros in the dictionary
                        print(f'{cls} is missing from the prediction!')

            # Update overall metrics
            n_images = len(images)
            scores_avg += batch_score/n_images
            iou_masks_avg += batch_iou_mask/n_images
            iou_boxes_avg += batch_iou_box/n_images

            # Update progress bar
            val_loop.set_postfix_str(f"Scores = {np.round(batch_score/n_images, 3)} | IoU mask, box = {np.round(batch_iou_mask/n_images, 3)}, {np.round(batch_iou_box/n_images, 3)}")

        # Average metrics over loader length
        scores_avg /= self.n_val
        iou_masks_avg /= self.n_val
        iou_boxes_avg /= self.n_val

        # Update overall metrics
        self.metrics_val['score_kit'].append(scores_avg[0])
        self.metrics_val['score_membrane'].append(scores_avg[1])
        self.metrics_val['iou_mask_kit'].append(iou_masks_avg[0])
        self.metrics_val['iou_mask_membrane'].append(iou_masks_avg[1])
        self.metrics_val['iou_box_kit'].append(iou_boxes_avg[0])
        self.metrics_val['iou_box_membrane'].append(iou_boxes_avg[1])
        
        return iou_masks_avg, iou_boxes_avg


    def get_metrics(self):
        """
        Returns metrics as Pandas DataFrames ready to be plotted
        """

        metrics_val_df = pd.DataFrame.from_dict(self.metrics_val)
        metrics_val_df.index.name = 'epochs'

        metrics_train_df = pd.DataFrame.from_dict(self.metrics_train)
        metrics_train_df.index.name = 'epochs'

        return metrics_train_df, metrics_val_df

    def save_model(self, epoch_ndx, file_path, stamp, elapsed_time):
        """
        Save model state, parameters, and metrics.
        """
        state = {
            'init_arguments':{'config_file': self.config_file, 
                              'kit_id': self.kit_id, 
                              'transformation_train': self.transformation_train,
                              'transformation_val': self.transformation_val},
            'model_state': self.model.state_dict(),  # model's state
            'metrics_train': self.metrics_train,
            'metrics_val': self.metrics_val,
            'filenames_train': self.filenames_train,
            'filenames_val': self.filenames_val,
            'stamp': self.stamp,
            'epoch_ndx': epoch_ndx,
            'elapsed_time': elapsed_time
        }
        torch.save(state, file_path + stamp + '.state')


    def load_state(self, path, device):
        """
        Load model state and metrics
        """
        state = torch.load(path, map_location=torch.device(device))
        # Initialize class with loaded arguments
        self.__init__(**state['init_arguments'])
        # Update attributes to be the ones of the loaded model
        self.stamp = state['stamp']
        self.metrics_train = state['metrics_train']
        self.metrics_val = state['metrics_val']
        self.filenames_train = state['filenames_train']
        self.filenames_val = state['filenames_val']
        
        # Load training and validation datasets and dataloaders
        self.loader_train, self.loader_val = self.init_dataloader()
        self.n_train = len(self.loader_train)
        self.n_val = len(self.loader_val)

        # Load model's state
        self.model.load_state_dict(state['model_state'])