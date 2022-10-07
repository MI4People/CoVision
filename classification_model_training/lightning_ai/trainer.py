import pytorch_lightning as pl
from argparse import ArgumentParser, Namespace

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from models.classification_models import ClassificationNet

from utils.data import DataModuleClassification

from callbacks.val_callback import LogPredictionsCallback

def main(hparams: Namespace):

    wandb_logger = WandbLogger(name=hparams.run_name, project=hparams.project_name, log_model="True")

    model = ClassificationNet(wandb_logger)

    dm = DataModuleClassification(path_to_train=hparams.train_data_path, path_to_test=hparams.test_data_path, batch_size=hparams.batch_size, load_size=hparams.load_size)

    checkpoint_callback = ModelCheckpoint(save_top_k=2, dirpath=hparams.checkpoint_dir, monitor="valid_acc_epoch", mode="max") # saves top-K checkpoint based on metric defined with monitor
    
    trainer = pl.Trainer(callbacks=checkpoint_callback, accelerator='gpu', devices=1, logger=wandb_logger, max_epochs=hparams.max_epochs)

    wandb_logger.watch(model)

    trainer.fit(model, dm)

    if hparams.test_data_path is not None:
        trainer.test(ckpt_path="best", datamodule=dm)



if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint_dir", type=str, help="path where the checkpoint (model etc.) should be saved")
    parser.add_argument("--run_name", type=str, help="Name of the run in wandb")
    parser.add_argument("--project_name", type=str, help="Name of the project in wandb")
    parser.add_argument("--train_data_path", type=str, help="path where the train dataset is stored")
    parser.add_argument("--test_data_path", default=None, type=str, help="path where the test target dataset is stored")
    parser.add_argument("--load_size", type=int, help="size to which images will get resized")
    parser.add_argument("--max_epochs", default=150, type=int, help="maximum epochs for training")
    parser.add_argument("--batch_size", default=16, type=int, help="batch size for training")
    hparams = parser.parse_args()

    main(hparams)