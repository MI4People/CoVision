import cv2
import torch
import torch.utils.data as data
from PIL import Image
import os
import numpy as np
import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning.trainer.supporters import CombinedLoader

from torchvision import transforms

from torchvision import transforms as TR

import albumentations as A
from albumentations.pytorch import ToTensorV2


class DataModuleClassification(pl.LightningDataModule):
    def __init__(self, path_to_train=None, path_to_test=None, batch_size=16, load_size: int = None):
        """
        Args:
            path_to_train
            path_to_test
            batch_size (int): Batch size used for train, val, test, metrics calculation
            load_size (int): Size to which the images are rescaled - will be squared image
        """
        super().__init__()

        self.path_to_train = path_to_train

        self.path_to_test = path_to_test

        self.load_size = load_size
        self.batch_size = batch_size

        self.train_aug = transforms.Compose([
            transforms.Resize((self.load_size, self.load_size)),
            transforms.ToTensor(),
        ])

        self.test_aug = transforms.Compose([
            transforms.Resize((self.load_size, self.load_size)),
            transforms.ToTensor(),
        ])

    def setup(self, stage=None):

        train_data = CustomDataset(self.path_to_train, augmentations=self.train_aug)
        train_set_size = int(len(train_data) * 0.8)
        val_set_size = len(train_data) - train_set_size
        self.train_data, self.val_data = data.random_split(train_data, [train_set_size, val_set_size])   

        if self.path_to_test is not None:
            self.test_data = CustomDataset(self.path_to_test, augmentations=self.test_aug)

    def train_dataloader(self):

        loader = data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=2, drop_last=True) # drop_last=True is needed to calculate the FFT for FDA properly

        return loader


    def val_dataloader(self):

        loader = data.DataLoader(self.val_data, batch_size=self.batch_size, num_workers=2)
        
        return loader

    def test_dataloader(self):

        loader_target_test = data.DataLoader(self.test_data, batch_size=self.batch_size, num_workers=2)

        return loader_target_test


class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, path_to_data, augmentations = None, backend = "pil"):
        """
        path_to_gt: path_to_groundtruth
        targets: numpy array of targets e.g. [1 0 1 1 0 0]
        augmentations: albumentations augmentations
        backend: the way the images are loaded - next to pil, one could add cv2
        channel: number of channels of the images from image_paths (default: 1)
        Issues:
            - add a resizing block
            - add more way to load images

        """

        self.path_to_data = path_to_data

        self.df = pd.read_csv(os.path.join(self.path_to_data, "gt.csv"))

        train_images = self.df.image.values.tolist()
        self.image_paths = [os.path.join(self.path_to_data, "images", i) for i in train_images]
        self.targets = self.df.target.values

        self.augmentations = augmentations
        self.backend = backend

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        target = self.targets[item]
        
        if self.backend == "pil":
            
            image = Image.open(self.image_paths[item])

            if self.augmentations is not None:
                image = self.augmentations(image)
                # image = augmented["image"]
        else:
            raise Exception("Backend not implemented")

        return image, target