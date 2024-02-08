"""
This file is used to create and organize data files in the right format so then they can be used for training a segmentation model.

It assumes all data is present on a folder whose path is specified in a configuration file (data_dir)
Inside this folder, the minimal requirement is the presence of two folders with names {kit_id}_images and {kit_id}_masks. 
The former must contain images with .jpg extension and the latter their corresponding masks in .png format 
(green pixels for membrane, red for kit, blue for background). 

Train and Test splits are created by the split_files_train_test() function, which creates 
{kit_id}_train_images, {kit_id}_test_images folders and their corresponding mask folders. 

Once train and test folders are created, we can build the target variables (ground truths), 
as they are required for training the segmentation model, from each human-annotated mask 
using the function `build_target_from_mask()`. For each RGB mask, this function returns 
a dictionary containing the class category masks (one-hot-encoding), bounding boxes, 
and labels, all in numpy array format.
"""

import yaml
import os
import sys
import random
import shutil
import numpy as np
import cv2

# Read the configuration file and use it as global variable all over the file
config_path = 'config_segmentation.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)['DataSettings']

def split_files_train_test(kit_id):
    """
    Create random train and test splits from base 'images' and 'masks' paths.
    Image and mask paths must follow the pattern {kit_id}_image and {kit_id}_masks. 
    Images must be .jpg and masks .png. It produces four files of the form 
    {kit_id}_train_images, {kit_id}_test_images, {kit_id}_train_masks, {kit_id}_test_masks, in
    the same folder as the original dataset, only if non of them already exist!

    Args:
        kit_id (string): name of the test kit (eg. btnx, abbott, etc).
    """

    images_path = os.path.join(config['data_dir'], f'{kit_id}_images')
    masks_path = os.path.join(config['data_dir'], f'{kit_id}_masks')
    
    images, masks = os.listdir(images_path), os.listdir(masks_path)

    ids = [os.path.splitext(f)[0] for f in images]
    ids_masks = [os.path.splitext(f)[0] for f in masks]

    # Make sure images and masks match
    assert set(ids) == set(ids_masks)

    # Make sure all images are .jpg and all masks .png
    assert all([f.endswith('.jpg') for f in images]) and all([f.endswith('.png') for f in masks])

    # Train-test split
    random.shuffle(ids)
    train_ids, test_ids = ids[: int(config['split_ratio'] * len(ids))], ids[int(config['split_ratio'] * len(ids)): ]
    assert len(train_ids) > len(test_ids)

    # Train and test paths
    train_images_path = os.path.join(config['data_dir'], f'{kit_id}_train_images')
    train_masks_path = os.path.join(config['data_dir'], f'{kit_id}_train_masks')
    test_images_path = os.path.join(config['data_dir'], f'{kit_id}_test_images')
    test_masks_path = os.path.join(config['data_dir'], f'{kit_id}_test_masks')

    for p in [train_images_path, train_masks_path, test_images_path, test_masks_path]:
        if os.path.exists(p):
            raise SystemExit(f'{p} exists! Cancelling script run!')
        else:
            os.makedirs(p)

    for train_id in train_ids:
        shutil.copy(os.path.join(images_path, f'{train_id}.jpg'), os.path.join(train_images_path, f'{train_id}.jpg'))
        shutil.copy(os.path.join(masks_path, f'{train_id}.png'), os.path.join(train_masks_path, f'{train_id}.png'))

    for test_id in test_ids:
        shutil.copy(os.path.join(images_path, f'{test_id}.jpg'), os.path.join(test_images_path, f'{test_id}.jpg'))
        shutil.copy(os.path.join(masks_path, f'{test_id}.png'), os.path.join(test_masks_path, f'{test_id}.png'))


def load_valid_filepaths(kit_id, dataset, filenames=None):
    """
    Returns list of images, and masks filepaths, only if there is 100% correspondence between files
    """

    images_dir = os.path.join(config['data_dir'], f'{kit_id}_{dataset}_images')
    masks_dir = os.path.join(config['data_dir'], f'{kit_id}_{dataset}_masks')

    # Catch missing file case
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f'{images_dir} folder does not exist! (Use split_files_train_test() to create train and test sets)')
    elif not os.path.exists(masks_dir):
        raise FileNotFoundError(f'{masks_dir} folder does not exist! (Use split_files_train_test() to create train and test sets)')

    
    image_paths = []
    mask_paths = []
    
    # If filenames are not specified, all filenames in folder are retrieved
    if filenames is None:
        filenames = sorted([name.replace('.jpg', '') for name in os.listdir(images_dir)])
    
    for file in filenames:
        
        # Image files
        image_path = os.path.join(images_dir, file + '.jpg')
        if os.path.exists(image_path):
            image_paths.append(image_path)
        else:
            raise Exception("Missing image file")
        # Mask files
        mask_path = os.path.join(masks_dir, file + '.png')
        if os.path.exists(mask_path):
            mask_paths.append(mask_path)
        else:
            raise Exception("Missing mask file")
    
    print(f'Loaded {len(filenames)} image and mask paths!')

    return image_paths, mask_paths


def compute_bounding_box_coordinates(mask):
                 
    all_coordinates = np.where(mask == 1)
    xmin = np.min(all_coordinates[1])
    ymin = np.min(all_coordinates[0])
    xmax = np.max(all_coordinates[1])
    ymax = np.max(all_coordinates[0])
    
    # Check that bounding box is consistent
    if xmax <= xmin or ymax <= ymin:
        raise ValueError('xmax < xmin OR ymax < ymin -> this should not be the case!')
    
    return xmin, ymin, xmax, ymax

def build_target_from_mask(mask):
    """
    Create category mask, bounding box, and labels from a given RGB mask.
    
    Args:
        mask (array (H, W, 3)): mask from where all other variables are computed

    Return:
        masks (array (2, H, W) [0-1]): binary masks of the kit and membrane
        boxes (array (2, 4)): coordinates of the bounding boxes in [xmin, ymin, xmax, ymax] format
        labels (array [2, 1]): labels for each class (1 for kit, 2 for membrane)
    """
    
    height, width = mask.shape[0:2]
    masks_cat = np.zeros([2, height, width])
    boxes = np.zeros([2, 4])
    labels = np.array(config['class_ids'])
    
    # Build category mask from RGB mask
    masks_cat[1] = np.all(mask == config['class_colors'][1], axis=-1).astype(int)  # Membrane
    # Include membrane mask to kit mask because, apparently, segmentation task is easier
    masks_cat[0] = np.all(mask == config['class_colors'][0], axis=-1).astype(int) + masks_cat[1] # Kit (+ Membrane)

    # Compute bounding box coordinates [xmin, ymin, xmax, ymax]
    boxes[0] = compute_bounding_box_coordinates(masks_cat[0])
    boxes[1] = compute_bounding_box_coordinates(masks_cat[1])

    return masks_cat, boxes, labels

