import torch
from torch import nn
import torchmetrics
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
from efficientnet_pytorch import EfficientNet

from sklearn import metrics

from torchvision.models import mobilenet_v3_small, efficientnet_v2_s

class ClassificationNet(pl.LightningModule):

    def __init__(
        self,
        wandb_logger,
        num_classes: int=4,
        lr: float = 1e-3,
):
        super().__init__()
        self.num_classes = num_classes
        self.lr = lr
        self.net = EfficientNet.from_pretrained("efficientnet-b2", in_channels = 3, num_classes = num_classes)
        # self.net = mobilenet_v3_small(num_classes=4)
        self.loss_function = nn.CrossEntropyLoss()

        # save hyper-parameters to self.hparams (auto-logged by W&B)
        self.save_hyperparameters()

        self.val_accuracy_torchmetrics = torchmetrics.Accuracy(num_classes=num_classes)
        self.confmat = torchmetrics.classification.MulticlassConfusionMatrix(num_classes=num_classes)

        self.confmat_test = torchmetrics.classification.MulticlassConfusionMatrix(num_classes=num_classes)

        self.wandb_logger = wandb_logger

        self.classes = ["positive", "negative", "empty", "invalid"]

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

        #######
        # Calculate different metrics here: (F1 Score, Precision, Recall, Sensitivity, Specifity etc.)

        # Shape of output: (B, C) with C= number of classes and B = batch size so e.g. (16,4)
        # , each value is a "probability" (after using nn.Softmax()) for the specific class e.g. output[0] = [7.1723e-01, 1.8264e-03, 2.7491e-01, 6.0351e-03]
        # Shape of target: (B) the value tells which class it is
        

        prediction = torch.argmax(output, dim=1)
        
        prediction = prediction.clone().cpu()
        target = target.clone().cpu()

        self.val_accuracy_torchmetrics.update(prediction, target)

        
        cuda_target = target.clone().to("cuda")
        self.confmat.update(output, cuda_target)

        return loss

    def validation_epoch_end(self, outputs):

        self.log('valid_acc_epoch', self.val_accuracy_torchmetrics.compute(), on_epoch=True)
        ## create an image from the conf matrix (A 4x4 tensor) and log the image to w&b

        self.wandb_logger.log_table(key="Confusion metric", columns=self.classes, data=self.confmat.compute().tolist())

        self.val_accuracy_torchmetrics.reset()
        self.confmat.reset()


    def test_step(self, batch, batch_idx):

        img, target = batch

        output = self(img)

        loss = self.loss_function(output, target)

        #######
        # Calculate different metrics here: (F1 Score, Precision, Recall, Sensitivity, Specifity etc.)

        cuda_target = target.clone().to("cuda")
        self.confmat_test.update(output, cuda_target)
        
        ############

        return loss

    def test_epoch_end(self, outputs):

        self.wandb_logger.log_table(key="Confusion metric test images", columns=self.classes, data=self.confmat_test.compute().tolist())
    
