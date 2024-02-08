import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Custom packages
from utils_segmentation.utils import collate_fn, compute_iou_mask, compute_iou_box
from utils_segmentation.visualization import show_images
from dataset_segmentation import LFASegmentationDataset

@torch.no_grad()
def run_inference_loader(loader, model, config_file, device=None):
    """
    Make predictions for all batches in loader.

    Returns:
        predictions_list: list of dictionaries of the form {masks, boxes} for each image in loader
        metrics_list: list of dictionaries of the form {scores, iou_masks, iou_boxes} for each image in loader
    """

    # Align input device with model's device
    if device is None:
        device = next(model.parameters()).device
    else:
        model.to(device)

    # List of dictionaries for predictions and metrics for the whole dataset in loader.
    predictions_list = []
    metrics_list = []

    for images, targets in loader:

        # Send images and targets to same device as model
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Inference step over single batch
        predictions_list_batch, metrics_list_batch = run_inference_batch(images, targets, model, config_file)

        # Update full output
        predictions_list += predictions_list_batch
        metrics_list += metrics_list_batch

    return predictions_list, metrics_list


@torch.no_grad()
def run_inference_batch(images, targets, model, config_file):
    """
    Make predictions for each batch of images, and select the masks, and boxes with the best scores for each class 
    and image. The predicted quantities are then compared with the targets to extract the mask and box IoU 
    for each class.

    Returns:
        predictions_list: list of dictionaries of the form {masks, boxes} for each image
        metrics_list: list of dictionaries of the form {scores, iou_masks, iou_boxes} for each image.
    """

    # Parameters from configuration file
    classes = config_file['DataSettings']['classes']
    n_classes = len(classes)
    class_ids = config_file['DataSettings']['class_ids']
    mask_thresholds = config_file['TrainingParameters']['mask_thresholds']

    # List of dictionaries for predictions and metrics for the whole batch.
    predictions_list = []
    metrics_list = []

    # Inference step
    predictions = model(images)

    for target, pred in zip(targets, predictions):

        # Get labels and scores, which are aligned
        labels = pred['labels'].tolist()
        scores = pred['scores'].tolist()
        assert len(labels) == len(scores)

        # Get boxes and masks, which are also naturally aligned
        boxes = pred['boxes']
        masks = pred['masks']
        assert boxes.shape[0] == masks.shape[0] == len(labels)

        # Keep track of best-score mask and boxes predictions for each image
        pred_dict = {'masks': torch.zeros_like(target['masks']),
                     'boxes': torch.zeros_like(target['boxes'])}

        # Keep track of best scores, and average IoUs for each image
        metrics_dict = {'scores': np.zeros(n_classes),
                        'iou_masks': np.zeros(n_classes),
                        'iou_boxes': np.zeros(n_classes)}

        # Loop over classes (i.e. kit and membrane)
        for i, cls in enumerate(classes):
            if class_ids[i] in labels:  # Check whether there is at least one prediction for that class

                # Get the maximum confidence class location (i.e. first occurrence in list)
                class_loc = labels.index(class_ids[i])

                # Get best class score
                class_score = scores[class_loc]

                # Get best class boxes and masks
                class_box = boxes[class_loc]
                class_mask = masks[class_loc, 0]

                # Binarize masks
                class_mask = (class_mask >= mask_thresholds[i]).to(torch.uint8)

                # Compute IoU
                class_iou_mask = compute_iou_mask(class_mask, target['masks'][i])
                class_iou_box = compute_iou_box(class_box, target['boxes'][i])

                # Update dictionaries
                pred_dict['masks'][i] = class_mask
                pred_dict['boxes'][i] = class_box

                metrics_dict['scores'][i] = class_score
                metrics_dict['iou_masks'][i] = class_iou_mask
                metrics_dict['iou_boxes'][i] = class_iou_box

            else:  # If there is no prediction, we leave all zeros in the dictionary
                print(f'{cls} is missing from the prediction!')

        # Send mask and boxes to cpu
        pred_dict['masks'] = pred_dict['masks'].to('cpu')
        pred_dict['boxes'] = pred_dict['boxes'].to('cpu')

        # Update lists
        predictions_list.append(pred_dict)
        metrics_list.append(metrics_dict)

    return predictions_list, metrics_list


def get_metrics(predictions, image_names):
    """
    Return metrics in the form of a Panda's dataframe with image_names as indices.
    """

    metrics_dict = {}

    for key in ['scores', 'iou_masks', 'iou_boxes']:
        for i, cls in enumerate(['kit', 'membrane']):
            metrics_dict[f'{key}_{cls}'] = [pred[key][i] for pred in predictions]

    metrics_df = pd.DataFrame.from_dict(metrics_dict)
    metrics_df.index = pd.Index(image_names, name='image_names')

    return metrics_df


def predict_testset(kit_id, config_file, model, save_filename=None, show_bool=True):
    """
    Run inference on all images in the kit_id test set and save all scores and IoU in a csv file.

    Returns:
        images_test: list of images in test set, as coming out from the Dataset class (i.e. after transformations)
        predictions_test: list of dictionaries containing the best-score masks, boxes, scores and IoUs
        metrics_test: list of dictionaries of the form {scores, iou_masks, iou_boxes} for each image.
        metrics_df: Pandas dataframe containing the scores and IoU for each image
    """

    data_settings = config_file['DataSettings']
    training_parameters = config_file['TrainingParameters']
    
    model.eval()

    # Dataset, dataloader and all images
    dataset_test = LFASegmentationDataset(data_settings, kit_id, dataset='test', transforms=None)

    loader_test = DataLoader(dataset=dataset_test,
                            batch_size=training_parameters['batch_size'],
                            shuffle=False,
                            num_workers=training_parameters['num_workers'],
                            collate_fn=collate_fn,
                            pin_memory=True)

    images_test = [img.to('cpu') for img, _ in dataset_test]

    # Run inference on all test data. Extract predictions and metrics
    predictions_test, metrics_test = run_inference_loader(loader_test, model, config_file)

    # Show all test images and their predictions
    if show_bool:
        show_images(images_test, predictions_test, metrics_test)

    # Format metrics as Dataframe
    metrics_df = get_metrics(metrics_test, image_names=dataset_test.filenames)
    
    # Save metrics to csv file
    if save_filename is not None:
        metrics_df.to_csv(os.path.join(data_settings['output_dir'], save_filename))

    return images_test, predictions_test, metrics_test, metrics_df


@torch.no_grad()
def run_inference(image, model, config_file):
    """
    Make prediction for a single raw image. No pre-procesing or ground truth required.

    Returns:
        best_masks:
        best_boxes:
        best_scores:
    """

    # Parameters from configuration file
    classes = config_file['DataSettings']['classes']
    n_classes = len(classes)
    class_ids = config_file['DataSettings']['class_ids']
    mask_thresholds = config_file['TrainingParameters']['mask_thresholds']

    # List of dictionaries for predictions and metrics for the whole batch.
    predictions_list = []
    metrics_list = []

    # Inference step (add input's batch dimension)
    prediction = model(image.unsqueeze(0))[0]

    # Get labels and scores, which are aligned
    labels = prediction['labels'].cpu().numpy().tolist()
    scores = prediction['scores'].cpu().numpy().tolist()
    assert len(labels) == len(scores)

    # Get boxes and masks, which are also naturally aligned
    boxes = prediction['boxes'].cpu().numpy()
    masks = prediction['masks'].cpu().numpy()
    assert boxes.shape[0] == masks.shape[0] == len(labels)

    # Store best-score, and corresponding mask and box prediction
    best_masks = np.zeros([n_classes, masks.shape[2], masks.shape[3]], dtype='uint8')
    best_boxes = np.zeros([n_classes, 4])
    best_scores = np.zeros(n_classes)

    # Loop over classes (i.e. kit and membrane)
    for i, cls in enumerate(classes):
        if class_ids[i] in labels:  # Check whether there is at least one prediction for that class

            # Get the maximum confidence class location (i.e. first occurrence in list)
            class_loc = labels.index(class_ids[i])

            # Get best class score
            class_score = scores[class_loc]

            # Get best class boxes and masks
            class_box = boxes[class_loc]
            class_mask = masks[class_loc, 0]

            # Binarize masks
            class_mask = (class_mask >= mask_thresholds[i]).astype(np.uint8)

            # Store 
            best_masks[i] = class_mask
            best_boxes[i] = class_box
            best_scores[i] = class_score

        else:  # If there is no prediction, we leave all zeros in the dictionary
            print(f'{cls} is missing from the prediction!')


    return best_masks, best_boxes, best_scores