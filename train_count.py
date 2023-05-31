"""
script to train on counting substructure tasks.
"""

from datasets.GraphCountDataset import GraphCountDatasetI2
import torch
import torch.nn as nn
from models.input_encoder import EmbeddingEncoder
import train_utils
import pytorch_lightning as pl
from interfaces.pl_model_interface import PlGNNTestonValModule
from interfaces.pl_data_interface import PlPyGDataTestonValModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Timer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import wandb
from torchmetrics import MeanAbsoluteError
import torch_geometric.transforms as T
from torch_geometric.data import Data


def add_node_feature(data: Data) -> Data:
    r"""Add identical initial node feature to all graphs.
    Arg:
        data (Data): PyG data.
    """
    data.x = torch.zeros([data.num_nodes, 1]).long()
    return data


def main():
    parser = train_utils.args_setup()
    parser.add_argument('--dataset_name', type=str, default="count_cycle", choices=("count_cycle", "count_graphlet"),
                        help='Name of dataset.')
    parser.add_argument('--task', type=int, default=0, choices=(0, 1, 2, 3, 4), help='Train task index.')
    parser.add_argument('--runs', type=int, default=3, help='Number of repeat run.')
    args = parser.parse_args()
    args = train_utils.update_args(args)

    path, pre_transform, follow_batch = train_utils.data_setup(args)

    train_dataset = GraphCountDatasetI2(root=path,
                                        dataname=args.dataset_name,
                                        split="train",
                                        pre_transform=T.Compose([add_node_feature, pre_transform]),
                                        transform=train_utils.PostTransform(args.wo_node_feature,
                                                                            args.wo_edge_feature,
                                                                            args.task))

    val_dataset = GraphCountDatasetI2(root=path,
                                      dataname=args.dataset_name,
                                      split="val",
                                      pre_transform=T.Compose([add_node_feature, pre_transform]),
                                      transform=train_utils.PostTransform(args.wo_node_feature,
                                                                          args.wo_edge_feature,
                                                                          args.task))

    test_dataset = GraphCountDatasetI2(root=path,
                                       dataname=args.dataset_name,
                                       split="test",
                                       pre_transform=T.Compose([add_node_feature, pre_transform]),
                                       transform=train_utils.PostTransform(args.wo_node_feature,
                                                                           args.wo_edge_feature,
                                                                           args.task))

    y_train_val = torch.cat([train_dataset.data.y, val_dataset.data.y], dim=0)
    mean = y_train_val.mean(dim=0)
    std = y_train_val.std(dim=0)
    train_dataset.data.y = (train_dataset.data.y - mean) / std
    val_dataset.data.y = (val_dataset.data.y - mean) / std
    test_dataset.data.y = (test_dataset.data.y - mean) / std

    for i in range(1, args.runs + 1):
        logger = WandbLogger(name=f'run_{str(i)}',
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

        loss_cri = nn.L1Loss()
        evaluator = MeanAbsoluteError()
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
                            ModelCheckpoint(monitor="val/metric", mode="min"),
                            LearningRateMonitor(logging_interval="epoch"),
                            timer
                        ]
                        )

        trainer.fit(modelmodule, datamodule=datamodule)
        val_result, test_result = trainer.test(modelmodule, datamodule=datamodule, ckpt_path="best")
        results = {"final/best_val_metric": val_result["val/metric"],
                   "final/best_test_metric": test_result["test/metric"],
                   "final/avg_train_time_epoch": timer.time_elapsed("train") / args.num_epochs,
                   }
        logger.log_metrics(results)
        wandb.finish()

    return


if __name__ == "__main__":
    main()

