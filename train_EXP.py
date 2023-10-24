"""
script to train on EXP task.
"""
import torch
import torch.nn as nn
from datasets.PlanarSATPairsDataset import PlanarSATPairsDataset
from models.input_encoder import EmbeddingEncoder
import train_utils
from interfaces.pl_model_interface import PlGNNTestonValModule
from interfaces.pl_data_interface import PlPyGDataTestonValModule
from lightning.pytorch import seed_everything
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, Timer
from lightning.pytorch.callbacks.progress import TQDMProgressBar
import torchmetrics
import wandb
from torch_geometric.data import Data
from torch_geometric.utils import sort_edge_index
import torch_geometric.transforms as T

def sort_pyg_edge_index(data: Data) -> Data:
    data.edge_index = sort_edge_index(data.edge_index)
    return data

def main():
    parser = train_utils.args_setup()
    parser.add_argument('--dataset_name', type=str, default="EXP", help='Name of dataset.')
    parser.add_argument('--folds', type=int, default=10, help='Number of fold in K-fold cross validation.')
    args = parser.parse_args()
    args = train_utils.update_args(args)

    path, pre_transform, follow_batch = train_utils.data_setup(args)

    dataset = PlanarSATPairsDataset(root=path,
                                    pre_transform=T.Compose([sort_pyg_edge_index, pre_transform]),
                                    transform=train_utils.PostTransform(args.wo_node_feature, args.wo_edge_feature))
    args.out_channels = dataset.num_classes

    for fold, (train_idx, test_idx, val_idx) in enumerate(
            zip(*train_utils.k_fold(dataset, args.folds, args.seed))):

        # Set random seed
        seed = train_utils.get_seed(args.seed)
        seed_everything(seed)

        train_dataset = dataset[train_idx]
        val_dataset = dataset[val_idx]
        test_dataset = dataset[test_idx]


        logger = WandbLogger(name=f'fold_{str(fold+1)}',
                             project=args.exp_name,
                             save_dir=args.save_dir,
                             offline=args.offline)
        logger.log_hyperparams(args)
        timer = Timer(duration=dict(weeks=4))

        datamodule = PlPyGDataTestonValModule(train_dataset=train_dataset,
                                              val_dataset=val_dataset,
                                              test_dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              num_workers=args.num_workers,
                                              follow_batch=follow_batch)
        loss_cri = nn.CrossEntropyLoss()
        evaluator = torchmetrics.classification.MulticlassAccuracy(num_classes=dataset.num_classes)
        init_encoder = EmbeddingEncoder(2, args.hidden_channels)

        modelmodule = PlGNNTestonValModule(loss_criterion=loss_cri,
                                           evaluator=evaluator,
                                           args=args,
                                           init_encoder=init_encoder)
        trainer = Trainer(accelerator="auto",
                          devices="auto",
                          max_epochs=args.num_epochs,
                          enable_checkpointing=True,
                          enable_progress_bar=True,
                          logger=logger,
                          callbacks=[TQDMProgressBar(refresh_rate=20),
                                     ModelCheckpoint(monitor="val/metric", mode="max"),
                                     LearningRateMonitor(logging_interval="epoch"),
                                     timer])

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
