# Based on:
# https://github.com/ray-project/ray/blob/master/python/ray/tune/examples/mnist_pytorch.py
from __future__ import print_function

import argparse
import os
import numpy as np
import random
import torch
import torch.nn as nn

import ray
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
from efficientnet_pytorch import EfficientNet
from utils.training_loops import training_loop
from utils.training_loops import val_loop
from train import get_data_loaders


def enable_reproducibility(seed, np, random, torch, os):
    """
    Set seed for reproducible training

    Args:
        seed (`int`): The seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


class TrainCoVision(tune.Trainable):
    def setup(self, config):
        # ray_train_torch.enable_reproducibility()
        enable_reproducibility(config.get("seed", 0), np, random, torch, os)

        use_cuda = config.get("args").use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        train_bs = config.get("train_bs", 16)
        val_bs = config.get("val_bs", 16)
        image_size = config.get("image_size", 224)

        gt = config.get("args").gt
        training_data_path = config.get("args").dataset

        fold = config.get("args").fold
        self.num_classes = config.get("args").num_classes

        self.train_loader, self.test_loader = get_data_loaders(
            gt, fold, training_data_path, train_bs, val_bs, image_size=image_size
        )

        self.model = EfficientNet.from_pretrained(
            config.get("model", "efficientnet-b2"),
            in_channels=3,
            num_classes=self.num_classes,
            dropout_rate=config.get("dropout_rate", 0.3),
            drop_connect_rate=config.get("drop_connect_rate", 0.2),
            batch_norm_momentum=config.get("batch_norm_momentum", 0.99),
            batch_norm_epsilon=config.get("batch_norm_epsilon", 1e-3),
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.get("lr", 1e-3),
            weight_decay=config.get("weight_decay", 0.0),
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5, threshold=0.0001, mode="max"
        )

        self.criterion = nn.CrossEntropyLoss()

    def step(self):
        (
            train_running_loss,
            train_num_correct,
            train_num_total,
            train_running_steps,
            train_f1,
        ) = training_loop(
            self.model,
            self.num_classes,
            self.device,
            self.train_loader,
            self.optimizer,
            self.criterion,
        )

        (
            test_running_loss,
            test_num_correct,
            test_num_total,
            test_running_steps,
            test_f1,
        ) = val_loop(
            self.model, self.num_classes, self.device, self.test_loader, self.criterion
        )

        self.scheduler.step(test_f1)

        return dict(
            train_acc=train_num_correct / train_num_total,
            train_loss=train_running_loss / train_running_steps,
            train_f1=train_f1,
            val_acc=test_num_correct / test_num_total,
            val_loss=test_running_loss / test_running_steps,
            val_f1=test_f1,
        )

    def save_checkpoint(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)

    def load_checkpoint(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        self.model.load_state_dict(torch.load(checkpoint_path))


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch CoVision")
    parser.add_argument(
        "--use-gpu", action="store_true", default=False, help="enables CUDA training"
    )
    parser.add_argument(
        "--ray-address", type=str, help="The Redis address of the cluster."
    )
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="path to the folder containing the images",
    )
    parser.add_argument(
        "--gt",
        type=str,
        required=True,
        help="path to the file that contains the gt csv file - see examples under /examples (needs to be added)",
    )
    parser.add_argument("--num_classes", type=int, default=4, help="number of classes")
    parser.add_argument(
        "--fold", type=int, default=1, help="which fold is the val fold"
    )

    args = parser.parse_args()
    ray.init(address=args.ray_address, num_cpus=6 if args.smoke_test else None)
    sched = ASHAScheduler(grace_period=10)

    np.random.seed(42)
    tuner = tune.Tuner(
        tune.with_resources(
            TrainCoVision, resources={"cpu": 4, "gpu": 0.3 if args.use_gpu else 0}
        ),
        run_config=train.RunConfig(
            stop={
                "val_f1": 0.95,
                "training_iteration": 3 if args.smoke_test else 30,
            },
            checkpoint_config=train.CheckpointConfig(
                checkpoint_score_attribute="val_f1",
                checkpoint_score_order="max",
                num_to_keep=5,
            ),
        ),
        tune_config=tune.TuneConfig(
            metric="val_f1",
            mode="max",
            scheduler=sched,
            num_samples=1 if args.smoke_test else 1,
        ),
        param_space={
            "args": args,
            # "lr": tune.loguniform(1e-4, 1e-3),
            "seed": 1,
            # "seed": tune.randint(0, 42),
            # "lr": tune.quniform(1e-4, 1e-3, 1e-4),
            # "weight_decay": tune.uniform(0.0, 1e-4),
            # "dropout_rate": tune.quniform(0.10, 0.4, 0.05),
            # "drop_connect_rate": tune.quniform(0.10, 0.4, 0.05),
            # "batch_norm_momentum": tune.choice([0.9, 0.997, 0.99]),
            # "batch_norm_epsilon": tune.choice([1e-3, 1e-5, 1e-6])
            # "momentum": tune.uniform(0.1, 0.9),
            # "model": tune.choice(
            #     ["efficientnet-b0", "efficientnet-b1", "efficientnet-b2"]
            # ),
            # "image_size": tune.choice([224, 240, 260]),
        },
    )
    results = tuner.fit()

    print("Best config is:", results.get_best_result().config)
