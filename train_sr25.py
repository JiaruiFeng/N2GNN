"""
script to train on SR25 task
"""

import torch
import torch.nn as nn
import os
import shutil
from datasets.SRDataset import SRDataset
from models.input_encoder import EmbeddingEncoder
import train_utils
import pytorch_lightning as pl
from interfaces.pl_model_interface import PlGNNTestonValModule
from interfaces.pl_data_interface import PlPyGDataTestonValModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Timer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import torchmetrics
import wandb



def main():
    parser = train_utils.args_setup()
    parser.add_argument('--dataset_name', type=str, default="sr25", help='Name of dataset.')
    args = parser.parse_args()
    args = train_utils.update_args(args)


    path, pre_transform, follow_batch = train_utils.data_setup(args)

    path = "data/" + args.dataset_name
    if os.path.exists(path + '/processed'):
        shutil.rmtree(path + '/processed')
    dataset = SRDataset(path,
                        pre_transform=pre_transform,
                        transform=train_utils.PostTransform(args.wo_node_feature, args.wo_edge_feature))
    dataset.data.x = dataset.data.x.long()
    dataset.data.y = torch.arange(len(dataset.data.y)).long()  # each graph is a unique class
    args.out_channels = len(dataset.data.y)
    train_dataset = dataset
    val_dataset = dataset
    test_dataset = dataset

    logger = WandbLogger(name=f'run',
                         project=args.exp_name,
                         save_dir=args.save_dir,
                         offline=args.offline)
    logger.log_hyperparams(args)
    timer = Timer(duration=dict(weeks=4))

    # Set random seed
    seed = train_utils.get_seed(args.seed)
    pl.seed_everything(seed)

    datamodule = PlPyGDataTestonValModule(train_dataset=train_dataset,
                                          val_dataset=val_dataset,
                                          test_dataset=test_dataset,
                                          batch_size=args.batch_size,
                                          num_workers=args.num_workers,
                                          follow_batch=follow_batch)

    loss_cri = nn.CrossEntropyLoss()
    evaluator = torchmetrics.classification.MulticlassAccuracy(num_classes=args.out_channels)
    init_encoder = EmbeddingEncoder(2, args.hidden_channels)


    modelmodule = PlGNNTestonValModule(loss_criterion=loss_cri,
                                       evaluator=evaluator,
                                       args=args,
                                       init_encoder=init_encoder)
    trainer = Trainer(
                    accelerator="auto",
                    devices="auto",
                    max_epochs=args.num_epochs,
                    enable_checkpointing=True,
                    enable_progress_bar=True,
                    logger=logger,
                    callbacks=[
                        TQDMProgressBar(refresh_rate=20),
                        ModelCheckpoint(monitor="val/metric", mode="max"),
                        LearningRateMonitor(logging_interval="epoch"),
                        timer
                    ]
                    )

    trainer.fit(modelmodule, datamodule)
    modelmodule.set_test_eval_still()
    val_result, test_result = trainer.validate(modelmodule, datamodule, ckpt_path="best")
    results = {"final/best_val_metric": val_result["val/metric"],
               "final/best_test_metric": test_result["test/metric"],
               "final/avg_train_time_epoch": timer.time_elapsed("train") / args.num_epochs,
               }
    logger.log_metrics(results)
    wandb.finish()

    return


if __name__ == "__main__":
    main()
