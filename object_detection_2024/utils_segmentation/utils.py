"""
File containing various functions used along the way in the training pipeline.
"""

import numpy as np
import random
import torch
import cv2

def set_seed(seed):
    """
    Set seed for numpy, torch, and all other random processes.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print(f"Random seed set as {seed}")

def resize_image(image, new_height):
    """
    Resize image to a a new height, preserving the original width/height ratio.
    """
    ratio = image.shape[1] / image.shape[0]
    new_width = int(new_height*ratio)
    image = cv2.resize(image, (new_width, new_height))
    return image

def collate_fn(batch):
    """
    Sets format in which batches are retrieved
    """
    return tuple(zip(*batch))

def compute_iou_mask(mask1, mask2):
    """
    Compute intersection over union (IoU) between masks. 
    It assumes binary masks containing only 1s (positive) and 0s (negative)
    """
    
    assert (mask1.unique().tolist() in [[0], [1], [0, 1]]) and (mask2.unique().tolist() in [[0], [1], [0, 1]]), 'Masks must contain 1s and 0s only!'
    
    intersection = mask1*mask2
    union = mask1 + mask2 - intersection
    iou = intersection.sum()/union.sum()
    return iou.item()

def compute_iou_box(box_cord_1, box_cord_2):
    """
    Compute intersection over union (IoU) between boxes. 
    It assumes box coordinates come in the format [xmin, ymin, xmax, ymax]
    """
    xmin_1, ymin_1, xmax_1, ymax_1 = box_cord_1
    box_area_1 = (xmax_1 - xmin_1) * (ymax_1 - ymin_1)
    assert box_area_1 > 0, "Area of Box 1 is negative, wrong box coordinates format"
    
    xmin_2, ymin_2, xmax_2, ymax_2 = box_cord_2
    box_area_2 = (xmax_2 - xmin_2) * (ymax_2 - ymin_2)
    assert box_area_2 > 0, "Area of Box 2 is negative, wrong box coordinates format"
    
    # Build coordinates for "intersection box"
    xmin_int = max(xmin_1, xmin_2)
    ymin_int = max(ymin_1, ymin_2)
    xmax_int = min(xmax_1, xmax_2)
    ymax_int = min(ymax_1, ymax_2)
    
    # Returns zero if one of the sides of the box is negative (i.e. no intersection!)
    intersection = max(xmax_int - xmin_int, 0) * max(ymax_int - ymin_int, 0)
    union =  box_area_1 + box_area_2 - intersection
    iou = intersection/union
    return iou.item()