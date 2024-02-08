"""
File containg the main Dataset class for image segmentation. 
"""

import os
import numpy as np
import torch
import cv2
from torchvision.transforms import functional as F

# Custom packages
from utils_segmentation.utils_dataset import load_valid_filepaths, build_target_from_mask
from transformations_segmentation import resize_image

class LFASegmentationDataset:
    def __init__(self, config, kit_id, dataset, filenames=None, transforms=None):
        
        # Configuration file and relevant features
        self.config = config
        self.data_dir = self.config['data_dir']
        self.resize_h = self.config['resize_height']
        self.kit_id = kit_id
        self.dataset = dataset
        assert self.dataset in ['train', 'test'], "dataset must be 'train' or 'test!"
        
        # Transformations
        self.transforms = transforms
        
        # If filenames not specified, load all filenames in folder
        if filenames is None:
            images_path = os.path.join(self.data_dir, f'{kit_id}_{dataset}_images')
            self.filenames = sorted([path.replace('.jpg', '') for path in os.listdir(images_path)])
        else:
            self.filenames = filenames

        # Load image, and mask full filepaths
        self.image_paths, self.mask_paths = load_valid_filepaths(self.kit_id, self.dataset, self.filenames)
        
    def __len__(self):
        
        return len(self.filenames)
    
    def __getitem__(self, idx):
        
        # Get corresponding image and mask path
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # Read image and mask as NumPy arrays
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path)
        
        # Resize image (excluded from transformations because it is mandatory for efficiency)
        image = resize_image(image, self.resize_h)
        mask = resize_image(mask, self.resize_h)

        # Check that image and masks have the same dimensions    
        assert image.shape[:2] == mask.shape[:2], "Image and Masks have different dimensions!"
        
        # Apply transforms if applicable
        if self.transforms is not None:
            image, mask = self.transforms(image, mask)
        
        # Build category mask, bounding boxes, and labels from RGB mask
        masks_cat, boxes, labels =  build_target_from_mask(mask)

        # Convert everything to a torch.Tensor
        image_t = F.to_tensor(image)  # Also scales to [0, 1] range and brings channel to first position!
        masks_t = torch.as_tensor(masks_cat, dtype=torch.uint8)
        boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
        labels_t = torch.as_tensor(labels, dtype=torch.int64)

        # Build target with ground-truths and image information
        target = {'masks': masks_t, 
                  'boxes': boxes_t,
                  'labels': labels_t}
        
        return image_t, target

