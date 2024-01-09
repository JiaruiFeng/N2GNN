"""
script to train on ZINC task.
"""
import torch.cuda
import torch.nn as nn
import torchmetrics
import wandb
from lightning.pytorch import seed_everything
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, Timer
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from torch_geometric.datasets import ZINC
import train_utils
from interfaces.pl_data_interface import PlPyGDataTestonValModule
from interfaces.pl_model_interface import PlGNNTestonValModule
from models.input_encoder import EmbeddingEncoder


def main():
    parser = train_utils.args_setup()
    parser.add_argument('--dataset_name', type=str, default="ZINC", help='Name of dataset.')
    parser.add_argument('--runs', type=int, default=10, help='Number of repeat run.')
    parser.add_argument('--full', action="store_true", help="If true, run ZINC full." )
    args = parser.parse_args()
    args = train_utils.update_args(args)
    if args.full:
        args.exp_name = "full_" + args.exp_name
    path, pre_transform, follow_batch = train_utils.data_setup(args)

    train_dataset = ZINC(path,
                         subset=not args.full,
                         split="train",
                         pre_transform=pre_transform,
                         transform=train_utils.PostTransform(args.wo_node_feature, args.wo_edge_feature))

    val_dataset = ZINC(path,
                       subset=not args.full,
                       split="val",
                       pre_transform=pre_transform,
                       transform=train_utils.PostTransform(args.wo_node_feature, args.wo_edge_feature))

    test_dataset = ZINC(path,
                        subset=not args.full,
                        split="test",
                        pre_transform=pre_transform,
                        transform=train_utils.PostTransform(args.wo_node_feature, args.wo_edge_feature))

    for i in range(1, args.runs + 1):
        logger = WandbLogger(name=f'run_{str(i)}',
                             project=args.exp_name,
                             save_dir=args.save_dir,
                             offline=args.offline)
        logger.log_hyperparams(args)
        timer = Timer(duration=dict(weeks=4))

        # Set random seed
        seed = train_utils.get_seed(args.seed)
        seed_everything(seed)

        datamodule = PlPyGDataTestonValModule(train_dataset=train_dataset,
                                              val_dataset=val_dataset,
                                              test_dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              num_workers=args.num_workers,
                                              follow_batch=follow_batch)
        loss_cri = nn.L1Loss()
        evaluator = torchmetrics.MeanAbsoluteError()
        args.mode = "min"
        init_encoder = EmbeddingEncoder(28, args.hidden_channels)
        edge_encoder = EmbeddingEncoder(4, args.inner_channels)

        modelmodule = PlGNNTestonValModule(loss_criterion=loss_cri,
                                           evaluator=evaluator,
                                           args=args,
                                           init_encoder=init_encoder,
                                           edge_encoder=edge_encoder)
        trainer = Trainer(accelerator="auto",
                          devices="auto",
                          max_epochs=args.num_epochs,
                          enable_checkpointing=True,
                          enable_progress_bar=True,
                          logger=logger,
                          callbacks=[TQDMProgressBar(refresh_rate=20),
                                     ModelCheckpoint(monitor="val/metric", mode=args.mode),
                                     LearningRateMonitor(logging_interval="epoch"),
                                     timer])

        trainer.fit(modelmodule, datamodule=datamodule)
        val_result, test_result = trainer.test(modelmodule, datamodule=datamodule, ckpt_path="best")
        results = {"final/best_val_metric": val_result["val/metric"],
                   "final/best_test_metric": test_result["test/metric"],
                   "final/avg_train_time_epoch": timer.time_elapsed("train") / args.num_epochs,
                   }
        print("torch.cuda.max_memory_reserved: %fGB" % (torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024))
        logger.log_metrics(results)
        wandb.finish()

    return


if __name__ == "__main__":
    main()
