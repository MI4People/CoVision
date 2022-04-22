import torch
import cv2
from torch.utils.data import Dataset, DataLoader

import numpy as np 

from PIL import Image
from PIL import ImageFile


class ClassificationDataset():

    def __init__(self, image_paths, targets, augmentations = None, backend = "pil"):
        """
        image_paths: list of paths to images
        targets: numpy array of targets e.g. [1 0 1 1 0 0]
        augmentations: albumentations augmentations
        backend: the way the images are loaded - next to pil, one could add cv2
        channel: number of channels of the images from image_paths (default: 1)
        Issues:
            - add a resizing block
            - add more way to load images

        """
        self.image_paths = image_paths
        self.targets = targets
        self.augmentations = augmentations
        self.backend = backend
        #self.channel = 1

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        target = self.targets[item]
        
        if self.backend == "pil":
            image = Image.open(self.image_paths[item])
            image = np.array(image)
            image = np.array([image])
            if self.augmentations is not None:
                augmented = self.augmentations(image=image)
                image = augmented["image"]
        else:
            raise Exception("Backend not implemented")

        return torch.tensor(image), torch.tensor(target)