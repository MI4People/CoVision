import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
from efficientnet_pytorch import EfficientNet

from sklearn import metrics

from torchvision.models import mobilenet_v3_small, efficientnet_v2_s

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
        # self.net = mobilenet_v3_small(num_classes=4)
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

        #######
        # Calculate different metrics here: (F1 Score, Precision, Recall, Sensitivity, Specifity etc.)

        # Shape of output: (B, C) with C= number of classes and B = batch size so e.g. (16,4)
        # , each value is a "probability" (after using nn.Softmax()) for the specific class e.g. output[0] = [7.1723e-01, 1.8264e-03, 2.7491e-01, 6.0351e-03]
        # Shape of target: (B) the value tells which class it is
        
        prediction = torch.argmax(output, dim=1)
        
        prediction = prediction.clone().cpu()
        target = target.clone().cpu()

        accuracy = metrics.accuracy_score(target, prediction)
        conf_mat = metrics.confusion_matrix(target, prediction)

        precision_micro = metrics.precision_score(target, prediction, average='micro')
        precision_macro = metrics.precision_score(target, prediction, average='macro')
        recall_micro = metrics.recall_score(target, prediction, average='micro')
        recall_macro = metrics.recall_score(target, prediction, average='macro')


        ############

        self.log('Val - CrossEntropyLoss', loss, prog_bar=True)
        self.log('Accuracy', accuracy, prog_bar=True)
        self.log('Precision micro', precision_micro, prog_bar=True)
        self.log('Precision macro', precision_macro, prog_bar=True)
        self.log('Recall micro', recall_micro, prog_bar=True)
        self.log('Recall macro', recall_macro, prog_bar=True)
        # self.log('Conf_mat', conf_mat, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):

        img, target = batch

        output = self(img)

        loss = self.loss_function(output, target)

        #######
        # Calculate different metrics here: (F1 Score, Precision, Recall, Sensitivity, Specifity etc.)

        print()
        print("shape of output: ", output.shape())
        print()
        print("shape of target: ", target.shape())
        print()


        ############

        self.log('Test - CrossEntropyLoss', loss, prog_bar=True)

        return loss
    
