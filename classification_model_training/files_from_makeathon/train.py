import os
import torch
from torch import nn


import argparse
import pandas as pd
import numpy as np
import json
from datetime import datetime
import random
import shutil
from torchvision import transforms

from efficientnet_pytorch import EfficientNet
from utils.data import ClassificationDataset
from utils.training_loops import training_loop
from utils.training_loops import val_loop


def get_data_loaders(
    gt: str,
    fold: int,
    training_data_path: str,
    train_bs: int,
    val_bs: int,
    image_size: int = 224,
):
    df = pd.read_csv(gt)

    df_train = df[df.fold != fold].reset_index(drop=True)
    df_val = df[df.fold == fold].reset_index(drop=True)

    # Set up the train_loader and val_loader
    train_aug = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation((0, 360)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_aug = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_images = df_train.image.values.tolist()
    train_images = [os.path.join(training_data_path, i) for i in train_images]
    train_targets = df_train.target.values

    val_images = df_val.image.values.tolist()
    val_images = [os.path.join(training_data_path, i) for i in val_images]
    val_targets = df_val.target.values

    train_dataset = ClassificationDataset(
        image_paths=train_images, targets=train_targets, augmentations=train_aug
    )

    val_dataset = ClassificationDataset(
        image_paths=val_images, targets=val_targets, augmentations=val_aug
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_bs, shuffle=True, num_workers=2
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=val_bs, shuffle=False, num_workers=2
    )
    return train_loader, val_loader


def train(
    model_name,
    image_size,
    fold,
    training_data_path,
    gt,
    num_classes,
    metric,
    device,
    epochs,
    train_bs,
    val_bs,
    outdir,
    lr,
    weight_decay,
    dropout_rate,
    drop_connect_rate,
    batch_norm_momentum,
    batch_norm_epsilon,
    pretrained_on_ImageNet,
    pretrained_own=None,
    starting_time=None,
):

    train_loader, val_loader = get_data_loaders(
        gt, fold, training_data_path, train_bs, val_bs, image_size
    )

    if pretrained_on_ImageNet:
        print("Using on ImageNet pretrained model")
        model = EfficientNet.from_pretrained(
            model_name,
            in_channels=3,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            drop_connect_rate=drop_connect_rate,
            batch_norm_momentum=batch_norm_momentum,
            batch_norm_epsilon=batch_norm_epsilon,
        )
        # model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True, num_classes=1, force_reload=True)

    else:
        print("Using NOT pretrained model")
        model = EfficientNet.from_name(
            model_name,
            in_channels=3,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            drop_connect_rate=drop_connect_rate,
            batch_norm_momentum=batch_norm_momentum,
            batch_norm_epsilon=batch_norm_epsilon,
        )
        # model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=False, num_classes=1)

    model.to(device)

    # Set loss function
    if num_classes == 1:
        print("Use BCEWithLogitsLoss")
        loss_function = nn.BCEWithLogitsLoss()
    else:
        print("Use CrossEntropyLoss")
        loss_function = nn.CrossEntropyLoss()

    # Set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Set scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, threshold=0.0001, mode="max"
    )

    best_score = 0

    # Start training
    for epoch in range(epochs):

        (
            train_running_loss,
            train_num_correct,
            train_num_total,
            train_running_steps,
            train_f1,
        ) = training_loop(
            model,
            num_classes,
            device,
            train_loader,
            optimizer,
            loss_function,
        )

        (
            val_running_loss,
            val_num_correct,
            val_num_total,
            val_running_steps,
            val_f1,
        ) = val_loop(model, num_classes, device, val_loader, loss_function)

        train_loss = train_running_loss / train_running_steps
        # train_acc = train_num_correct / train_num_total
        val_loss = val_running_loss / val_running_steps
        val_acc = val_num_correct / val_num_total

        print(
            f"Epoch = {epoch+1}, train_loss = {train_loss}, train_f1 = {train_f1}, val_loss = {val_loss}, val_f1 = {val_f1}"
        )

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_f1)
        else:
            scheduler.step()

        is_best = val_f1 > best_score
        best_score = max(val_f1, best_score)

        torch.save(model.state_dict(), os.path.join(outdir, "model.pt"))
        if is_best:
            shutil.copyfile(
                os.path.join(outdir, "model.pt"), os.path.join(outdir, "model_best.pt")
            )

            if opt.metrics_file_path is not None:
                json.dump(
                    obj={
                        "f1_score": val_f1,
                        "accuracy": val_acc,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "epoch": epoch + 1,
                    },
                    fp=open(opt.metrics_file_path, "w"),
                    indent=4,
                )
            print("Model with improved f1_score saved to outdir")

    end_time = datetime.now()

    end_time = end_time.strftime("%H:%M:%S")
    print("Start Time = ", starting_time, "End Time = ", end_time)


def set_seed(seed):
    """
    Set seed for reproducible training

    Args:
        seed (`int`): The seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def enable_determinism():
    """
    https://pytorch.org/docs/stable/notes/randomness.html#reproducibility
    """
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=80, help="number of epochs")
    parser.add_argument(
        "--train_batch", type=int, default=16, help="batch size for training"
    )
    parser.add_argument(
        "--val_batch", type=int, default=16, help="batch size for validation"
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument(
        "--pretrained_own", type=str, help="path to a pretrained model (.bin)"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="f1_score",
        help="The metric on which to save improved models - (f1_score,auc, precision)",
    )
    # Needed inputs:
    parser.add_argument("--num_classes", type=int, default=1, help="number of classes")
    parser.add_argument(
        "--pretrained_on_ImageNet",
        default=False,
        action="store_true",
        help="Use a pretrained model",
    )
    parser.add_argument("--fold", type=int, help="which fold is the val fold")
    parser.add_argument(
        "--dataset", type=str, help="path to the folder containing the images"
    )
    parser.add_argument(
        "--gt",
        type=str,
        help="path to the file that contains the gt csv file - see examples under /examples (needs to be added)",
    )
    parser.add_argument(
        "--outdir", type=str, help="outputdir where the files are stored"
    )
    parser.add_argument("--seed", type=int, help="Seed value")
    parser.add_argument("--dropout_rate", type=float, default=0.3)
    parser.add_argument("--drop_connect_rate", type=float, default=0.2)
    parser.add_argument("--batch_norm_momentum", type=float, default=0.99)
    parser.add_argument("--batch_norm_epsilon", type=float, default=1e-3)
    parser.add_argument(
        "--metrics_file_path",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--model", default="efficientnet-b2", type=str, help="model name"
    )
    parser.add_argument("--image_size", default=224, type=int, help="image size")

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

    if opt.seed is not None:
        print("Setting seed =", opt.seed)
        set_seed(opt.seed)
        enable_determinism()

        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    train(
        opt.model,
        opt.image_size,
        opt.fold,
        opt.dataset,
        opt.gt,
        opt.num_classes,
        opt.metric,
        opt.device,
        opt.epochs,
        opt.train_batch,
        opt.val_batch,
        opt.outdir,
        opt.lr,
        opt.weight_decay,
        opt.dropout_rate,
        opt.drop_connect_rate,
        opt.batch_norm_momentum,
        opt.batch_norm_epsilon,
        opt.pretrained_on_ImageNet,
        opt.pretrained_own,
        starting_time,
    )
