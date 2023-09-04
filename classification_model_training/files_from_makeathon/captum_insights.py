#!/usr/bin/env python
# coding: utf-8
#
# Based in https://captum.ai/tutorials/CIFAR_TorchVision_Captum_Insights

import argparse
import os

import pandas as pd
import torch

import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet

from captum.insights import AttributionVisualizer, Batch
from captum.insights.attr_vis.features import ImageFeature

from utils.data import ClassificationDataset


def get_classes():
    classes = [
        "positive",
        "negative",
        "empty",
        "invalid",
    ]
    return classes


def get_pretrained_model(single_model_path: str, num_classes: int, device: str):
    model = EfficientNet.from_name(
        "efficientnet-b2", in_channels=3, num_classes=num_classes
    )
    model.load_state_dict(torch.load(single_model_path))
    model.to(device)
    return model


def baseline_func(input):
    return input * 0


def load_data(test_data_path, gt):

    df_test = pd.read_csv(gt)

    test_images = df_test.image.values.tolist()
    test_images = [os.path.join(test_data_path, i) for i in test_images]
    test_targets = df_test.target.values

    return test_images, test_targets


def formatted_data_iter(dataset_path: str, gt: str, val_bs: int, device: str):
    test_images, test_targets = load_data(dataset_path, gt)
    prediction_aug = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    dataset = ClassificationDataset(
        image_paths=test_images, targets=test_targets, augmentations=prediction_aug
    )
    dataloader = iter(
        torch.utils.data.DataLoader(
            dataset, batch_size=val_bs, shuffle=False, num_workers=2
        )
    )
    while True:
        images, labels = next(dataloader)
        yield Batch(inputs=images.to(device), labels=labels.to(device))


def serve(
    dataset: str,
    gt: str,
    val_bs: int,
    single_model_path: str,
    num_classes: int,
    device: str,
    debug: bool = False,
):

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    model = get_pretrained_model(single_model_path, num_classes, device)
    visualizer = AttributionVisualizer(
        models=[model],
        score_func=lambda o: torch.nn.functional.softmax(o, 1),
        classes=get_classes(),
        features=[
            ImageFeature(
                "Photo",
                baseline_transforms=[baseline_func],
                input_transforms=[normalize],
            )
        ],
        dataset=formatted_data_iter(dataset, gt, val_bs, device),
    )

    visualizer.serve(debug=debug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to the folder containing the images size: 260x260",
    )
    parser.add_argument(
        "--gt",
        type=str,
        help="path to the file that contains the gt csv file - see examples under /examples (needs to be added)",
    )
    parser.add_argument(
        "--num_classes", type=int, help="The number of classes in the test dataset"
    )
    parser.add_argument(
        "--single_model_path",
        default=None,
        type=str,
        help="Use only for single model prediction: path to the folder containing the model file: .bin",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--val_bs", type=int, default=16)
    parser.add_argument("--debug", action="store_true")
    opt = parser.parse_args()

    serve(
        opt.dataset,
        opt.gt,
        opt.val_bs,
        opt.single_model_path,
        opt.num_classes,
        opt.device,
        opt.debug,
    )
