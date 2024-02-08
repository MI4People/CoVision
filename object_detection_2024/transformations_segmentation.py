"""
File containing data transformations to use during training and inference.
"""


import random
import cv2
import numpy as np
import torch
import albumentations as A


def resize_image(image, new_height):
    """
    Resize image to a a new height, preserving the original width/height ratio.
    """
    ratio = image.shape[1] / image.shape[0]
    new_width = int(new_height*ratio)
    image = cv2.resize(image, (new_width, new_height))
    return image


class TransformationSegmentationTraining():
    """
    Transformation for segmentation training dataset.
    """
    def __init__(self, parameters):
        self.param = parameters

        # Rotate
        self.rotate = A.Rotate(limit=self.param['rotate_limit'], p=self.param['rotate_p'], border_mode=cv2.BORDER_CONSTANT)
        # Horizontal flip
        self.horizontal_flip = A.HorizontalFlip(p=self.param['horizontal_flip_p'])
        # Blur
        self.blur = A.Blur(blur_limit=self.param['blur_limit'], p=self.param['blur_p'])
        # Color jitter
        self.color_jitter = A.ColorJitter(brightness=self.param['color_jitter_brightness'], 
                                        contrast=self.param['color_jitter_contrast'],
                                        saturation=self.param['color_jitter_saturation'], 
                                        p=self.param['color_jitter_p'])
        # Composite transformation
        self.transform = A.Compose([self.rotate, 
                                    self.horizontal_flip, 
                                    A.OneOf([
                                        self.blur, 
                                        self.color_jitter], p=1.0)
                                    ])

    def __call__(self, image, mask):
        
        transform_dict = self.transform(image=image, mask=mask)

        return transform_dict['image'], transform_dict['mask']
        
