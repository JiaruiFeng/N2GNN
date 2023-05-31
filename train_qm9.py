"""
script to train on QM9 targets.
"""

import torch
import torch.nn as nn
from torch import Tensor
from datasets.QM9Dataset import QM9, conversion
from models.input_encoder import QM9InputEncoder, EmbeddingEncoder
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
from torchmetrics.functional.regression.mae import _mean_absolute_error_compute
import torch_geometric.transforms as T
from torch_geometric.data import Data


class InputTransform(object):
    """QM9 input feature transformation. Concatenate x and z together.
    """
    def __init__(self):
        super().__init__()

    def __call__(self,
                 data: Data) -> Data:
        x = data.x
        z = data.z
        data.x = torch.cat([z.unsqueeze(-1), x], dim=-1)
        data.edge_attr = torch.where(data.edge_attr == 1)[-1]
        return data


class MeanAbsoluteErrorQM9(MeanAbsoluteError):
    def __init__(self,
                 std,
                 conversion,
                 **kwargs):
        super().__init__(**kwargs)
        self.std = std
        self.conversion = conversion

    def compute(self) -> Tensor:
        return (_mean_absolute_error_compute(self.sum_abs_error, self.total) * self.std) / self.conversion


def main():
    parser = train_utils.args_setup()
    parser.add_argument('--dataset_name', type=str, default="QM9", help='Name of dataset.')
    parser.add_argument('--task', type=int, default=11, choices=list(range(19)), help='Train target.')
    parser.add_argument('--search', action="store_true", help="If true, run all first 12 targets.")

    args = parser.parse_args()
    args = train_utils.update_args(args, add_task=False)
    path, pre_transform, follow_batch = train_utils.data_setup(args)

    if args.search:
        for target in range(12):
            args.task = target
            dataset = QM9(path,
                          pre_transform=T.Compose([InputTransform(), pre_transform]),
                          transform=train_utils.PostTransform(args.wo_node_feature,
                                                              args.wo_edge_feature,
                                                              args.task
                                                              ))

            dataset = dataset.shuffle()

            tenprecent = int(len(dataset) * 0.1)
            mean = dataset.data.y[tenprecent:].mean(dim=0)
            std = dataset.data.y[tenprecent:].std(dim=0)
            dataset.data.y = (dataset.data.y - mean) / std

            train_dataset = dataset[2 * tenprecent:]
            test_dataset = dataset[:tenprecent]
            val_dataset = dataset[tenprecent:2 * tenprecent]

            logger = WandbLogger(name=f'target_{str(args.task + 1)}',
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

            loss_cri = nn.MSELoss()
            evaluator = MeanAbsoluteErrorQM9(std[args.task].item(), conversion[args.task].item())
            init_encoder = QM9InputEncoder(args.hidden_channels)
            edge_encoder = EmbeddingEncoder(4, args.hidden_channels)


            modelmodule = PlGNNTestonValModule(loss_criterion=loss_cri,
                                               evaluator=evaluator,
                                               args=args,
                                               init_encoder=init_encoder,
                                               edge_encoder=edge_encoder)
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

    else:

        dataset = QM9(path,
                      pre_transform=T.Compose([InputTransform(), pre_transform]),
                      transform=train_utils.PostTransform(args.wo_node_feature,
                                                          args.wo_edge_feature,
                                                          args.task
                                                          ))

        dataset = dataset.shuffle()

        tenprecent = int(len(dataset) * 0.1)
        mean = dataset.data.y[tenprecent:].mean(dim=0)
        std = dataset.data.y[tenprecent:].std(dim=0)
        dataset.data.y = (dataset.data.y - mean) / std
        train_dataset = dataset[2 * tenprecent:]
        test_dataset = dataset[:tenprecent]
        val_dataset = dataset[tenprecent:2 * tenprecent]

        logger = WandbLogger(name=f'target_{str(args.task + 1)}',
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

        loss_cri = nn.MSELoss()
        evaluator = MeanAbsoluteErrorQM9(std[args.task].item(), conversion[args.task].item())
        init_encoder = QM9InputEncoder(args.hidden_channels)
        edge_encoder = EmbeddingEncoder(4, args.hidden_channels)

        modelmodule = PlGNNTestonValModule(loss_criterion=loss_cri,
                                           evaluator=evaluator,
                                           args=args,
                                           init_encoder=init_encoder,
                                           edge_encoder=edge_encoder)
        trainer = Trainer(
            accelerator="auto",
            devices="auto",
            max_epochs=args.num_epochs,
            enable_checkpointing=True,
            enable_progress_bar=True,
            logger=logger,
            callbacks=[
                TQDMProgressBar(refresh_rate=5),
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


