import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
from efficientnet_pytorch import EfficientNet


class ClassificationNet(pl.LightningModule):

    def __init__(
        self,
        num_classes: int=4,
        lr: float = 1e-3,
):
        super().__init__()
        self.num_classes = num_classes
        self.lr = lr
        self.net = EfficientNet.from_pretrained("efficientnet-b2", in_channels = 3, num_classes = num_classes)
        self.loss_function = nn.CrossEntropyLoss()

        # save hyper-parameters to self.hparams (auto-logged by W&B)
        self.save_hyperparameters()

    def forward(self, x):

        return self.net(x)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        return opt

    def training_step(self, batch, batch_nb):

        img, target = batch

        output = self(img)

        loss = self.loss_function(output, target)

        self.log('Train - CrossEntropyLoss', loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_nb):

        img, target = batch

        output = self(img)

        loss = self.loss_function(output, target)

        #Calculate different metrics here: (F1 Score, Precision, Recall, Sensitivity, Specifity etc.)

        self.log('Val - CrossEntropyLoss', loss, prog_bar=True)

        return loss
    
