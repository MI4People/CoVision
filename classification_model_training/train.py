from configparser import Interpolation
import os
from sklearn import metrics
import torch
from torch import nn

import albumentations
import argparse
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime
import cv2
from torchvision import transforms

from efficientnet_pytorch import EfficientNet
from utils.data import ClassificationDataset
from utils.training_loops import training_loop
from utils.training_loops import val_loop
from utils.calc_mean_std import get_mean_std

def train(mean, std, fold, training_data_path, gt, num_classes, metric, device, epochs, train_bs, val_bs, outdir, lr, pretrained_on_ImageNet, pretrained_own=None):

    df = pd.read_csv(gt)
    mean = mean
    std = std


    df_train = df[df.fold != fold].reset_index(drop=True)
    df_val = df[df.fold == fold].reset_index(drop=True)


    print()
    print("use_pretrained: ", pretrained_on_ImageNet)
    print()

    if pretrained_on_ImageNet:
        print("Using on ImageNet pretrained model")
        model = EfficientNet.from_pretrained("efficientnet-b2", in_channels = 3, num_classes = num_classes)
        # model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True, num_classes=1, force_reload=True)
    
    else:
        print("Using NOT pretrained model")
        model = EfficientNet.from_name('efficientnet-b2', in_channels = 3, num_classes = num_classes)
        # model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=False, num_classes=1)
    
    model.to(device)

    # Set up the train_loader and val_loader
    train_aug = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_aug = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_images = df_train.image.values.tolist()
    train_images = [os.path.join(training_data_path, i) for i in train_images]
    train_targets = df_train.target.values

    val_images = df_val.image.values.tolist()
    val_images = [os.path.join(training_data_path, i) for i in val_images]
    val_targets = df_val.target.values

    train_dataset = ClassificationDataset(
        image_paths=train_images,
        targets=train_targets,
        augmentations=train_aug)

    val_dataset = ClassificationDataset(
        image_paths=val_images,
        targets=val_targets,
        augmentations=val_aug)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_bs,
        shuffle=True,
        num_workers=2
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_bs,
        shuffle=False,
        num_workers=2
    )

    # Set loss function
    if num_classes == 1:
        print("Use BCEWithLogitsLoss")
        loss_function = nn.BCEWithLogitsLoss()
    else:
        print("Use CrossEntropyLoss")
        loss_function = nn.CrossEntropyLoss()


    # Set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Set scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=5,
        threshold=0.0001,
        mode="max"
    )

    train_loss_all = []
    val_loss_list = []
    accuracy_list = []
    f1_score_list = []
    f1_score_for_model_saving_list = []
    auc_list = []

    # Start training
    for epoch in range(epochs):

        train_loss = training_loop(model, num_classes, device, train_loader, optimizer, loss_function, epoch, epochs)

        targets, predictions, accuracy, val_loss = val_loop(model, num_classes, device, val_loader, loss_function)

        assert np.array_equal(targets, val_targets), "targets from val_loop are not equal to val_targets (source of the validation data)"

        print()
        print("targets: ", targets)
        print("predictions: ", predictions)
        print()

        if num_classes == 1:
            predictions = np.vstack((predictions)).ravel()
            f1_score = metrics.f1_score(targets, predictions)
            accuracy = metrics.accuracy_score(targets, predictions)
            # auc = metrics.roc_auc_score(targets, predictions)

        else:
            f1_score = metrics.f1_score(targets, predictions, average='micro')
            #auc = metrics.roc_auc_score(targets, predictions, multi_class ="ovr")
            auc = "needs to be debugged"

        print(f"Epoch = {epoch+1}, accuracy = {accuracy}, F1_Score = {f1_score}")
    
        # Learning rate scheduler improves according to chosen metric
        if metric == "auc":
            scheduler.step(auc)
        
            if all(auc > i for i in auc_list):
                torch.save(model.state_dict(), os.path.join(outdir, f"model_fold_{fold}.bin"))
                print("Model with improved auc saved to outdir")

        elif metric == "f1_score":
            scheduler.step(f1_score)
        
            if all(f1_score > i for i in f1_score_for_model_saving_list) and epoch >= 3:
                torch.save(model.state_dict(), os.path.join(outdir, f"model_fold_{fold}_{epoch}.bin"))
                print("Model with improved f1_score saved to outdir")

        elif metric == "accuracy":
            scheduler.step(accuracy)
        
            if all(accuracy > i for i in f1_score_for_model_saving_list) and epoch >= 3:
                torch.save(model.state_dict(), os.path.join(outdir, f"model_fold_{fold}.bin"))
                print("Model with improved accuracy saved to outdir")

        if epoch >= 15:
            f1_score_for_model_saving_list.append(f1_score)
        train_loss_all.append(train_loss)
        val_loss_list.append(val_loss)
        accuracy_list.append(accuracy)
        f1_score_list.append(f1_score)


    # Make a function for the following lines
    train_loss_all = np.array(train_loss_all)
    train_loss_all = train_loss_all.flatten()
    train_loss_plot = plt.figure()
    plt.plot(train_loss_all)
    plt.xlabel("Step")
    plt.ylabel("Training Loss")
    train_loss_plot.savefig(os.path.join(opt.outdir, f'training_loss_fold{opt.fold}.png'))

    val_loss_plot = plt.figure()
    plt.plot(val_loss_list)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    val_loss_plot.savefig(os.path.join(opt.outdir, f'validation_loss_fold{opt.fold}.png'))

    accuracy_plot = plt.figure()
    plt.plot(accuracy_list)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    accuracy_plot.savefig(os.path.join(opt.outdir, f'accuracy_fold{opt.fold}.png'))

    f1_score_plot = plt.figure()
    plt.plot(f1_score_list)
    plt.xlabel("Epoch")
    plt.ylabel("F1_score")
    f1_score_plot.savefig(os.path.join(opt.outdir, f'f1_scores_fold{opt.fold}.png'))

    print("plots saved..")

    end_time = datetime.now()

    end_time = end_time.strftime("%H:%M:%S")
    print("Start Time = ", starting_time, "End Time = ", end_time)
        


if __name__ == "__main__" :

    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=80, help="number of epochs")
    parser.add_argument("--train_batch", type=int, default=16, help="batch size for training")
    parser.add_argument("--val_batch", type=int, default=16, help="batch size for validation")
    parser.add_argument("--lr", type=int, default=1e-3)
    parser.add_argument("--pretrained_own", type=str, help="path to a pretrained model (.bin)")
    parser.add_argument("--metric", type=str, default="f1_score", help="The metric on which to save improved models - (f1_score,auc, precision)")
    # Needed inputs:
    parser.add_argument("--num_classes", type=int, default=1, help="number of classes")
    parser.add_argument("--pretrained_on_ImageNet", default=False, action='store_true', help="Use a pretrained model")
    parser.add_argument("--fold", type=int, help="which fold is the val fold")
    parser.add_argument("--dataset", type=str, help="path to the folder containing the images")
    parser.add_argument("--gt", type=str, help="path to the file that contains the gt csv file - see examples under /examples (needs to be added)")
    parser.add_argument("--outdir", type=str, help="outputdir where the files are stored")

    opt = parser.parse_args()

    opt_dict = vars(opt)

    json_object = json.dumps(opt_dict)

    print("Numm classes before: ", opt.num_classes)
    if opt.num_classes == 2:
        opt.num_classes = 1
    print("Numm classes after: ", opt.num_classes)

    epoch = 0

    print(epoch)

    # open(os.path.join(opt.outdir, f'training_options{opt.fold}.txt'), 'w').close()

    # with open(os.path.join(opt.outdir, f'training_options{opt.fold}.txt'), 'a') as f:
    #     json.dump(json_object, f)
    
    starting_time = datetime.now()

    starting_time = starting_time.strftime("%H:%M:%S")
    print("Starting Time =", starting_time)

    mean = mean=[0.485, 0.456, 0.406]
    std = std=[0.229, 0.224, 0.225]

    train(mean, std, opt.fold, opt.dataset, opt.gt, opt.num_classes, opt.metric, opt.device, opt.epochs, opt.train_batch, opt.val_batch, opt.outdir, opt.lr, opt.pretrained_on_ImageNet, opt.pretrained_own)