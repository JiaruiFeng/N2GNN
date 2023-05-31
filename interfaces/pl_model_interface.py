"""
Pytorch lightning model module for PyG model.
"""

from argparse import ArgumentParser
from copy import deepcopy as c
from typing import Dict, List

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torchmetrics import Metric

from models.model_construction import make_model


class PlGNNModule(pl.LightningModule):
    r"""Basic pytorch lighting module for GNNs.
    Args:
        loss_criterion (nn.Module) : Loss compute module.
        evaluator (Metric): Evaluator for evaluating model performance.
        args (ArgumentParser): Arguments dict from argparser.
        init_encoder (nn.Module): Node feature initial encoder.
        edge_encoder (nn.Module): Edge feature encoder.
    """

    def __init__(self,
                 loss_criterion: nn.Module,
                 evaluator: Metric,
                 args: ArgumentParser,
                 init_encoder: nn.Module = None,
                 edge_encoder: nn.Module = None,
                 ):
        super(PlGNNModule, self).__init__()
        self.model = make_model(args, init_encoder, edge_encoder)
        self.loss_criterion = loss_criterion
        self.train_evaluator = c(evaluator)
        self.val_evaluator = c(evaluator)
        self.test_evaluator = c(evaluator)
        self.args = args

    def forward(self,
                data: Data) -> Tensor:
        return self.model(data)

    def training_step(self,
                      batch: Data,
                      batch_idx: Tensor) -> Dict:
        y = batch.y.squeeze()
        out = self.model(batch).squeeze()
        loss = self.loss_criterion(out, y)
        self.log("train/loss",
                 loss,
                 prog_bar=True,
                 batch_size=self.args.batch_size)
        self.train_evaluator.update(out, y)
        return {'loss': loss, 'preds': out, 'target': y}

    def on_train_epoch_end(self) -> None:
        self.log("train/metric",
                 self.train_evaluator.compute(),
                 prog_bar=False)
        self.train_evaluator.reset()

    def validation_step(self,
                        batch: Data,
                        batch_idx: Tensor) -> Dict:
        y = batch.y.squeeze()
        out = self.model(batch).squeeze()
        loss = self.loss_criterion(out, y)
        self.log("val/loss",
                 loss,
                 prog_bar=False,
                 batch_size=self.args.batch_size)
        self.val_evaluator.update(out, y)
        return {'loss': loss, 'preds': out, 'target': y}

    def on_validation_epoch_end(self) -> None:
        self.log("val/metric",
                 self.val_evaluator.compute(),
                 prog_bar=True)
        self.val_evaluator.reset()

    def test_step(self,
                  batch: Data,
                  batch_idx: Tensor) -> Dict:
        y = batch.y.squeeze()
        out = self.model(batch).squeeze()
        loss = self.loss_criterion(out, y)
        self.log("test/loss",
                 loss,
                 prog_bar=False,
                 batch_size=self.args.batch_size)
        self.test_evaluator.update(out, y)
        return {'loss': loss, 'preds': out, 'target': y}

    def on_test_epoch_end(self) -> None:
        self.log("test/metric",
                 self.test_evaluator.compute(),
                 prog_bar=True)
        self.test_evaluator.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=float(self.args.l2_wd),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, factor=self.args.factor, patience=self.args.patience, min_lr=self.args.min_lr
                ),
                "monitor": "val/metric",
                "frequency": 1,
                "interval": "epoch",
            },
        }

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        if 'v_num' in tqdm_dict:
            del tqdm_dict['v_num']
        return tqdm_dict


class PlGNNTestonValModule(PlGNNModule):
    r"""Given a preset evaluation interval, run test dataset during validation
        to have a snoop on test performance when meet the interval during .
    """

    def __init__(self,
                 loss_criterion: nn.Module,
                 evaluator: Metric,
                 args: ArgumentParser,
                 init_encoder: nn.Module = None,
                 edge_encoder: nn.Module = None,
                 ):
        super().__init__(loss_criterion, evaluator, args, init_encoder, edge_encoder)
        self.test_eval_still = self.args.test_eval_interval

    def validation_step(self,
                        batch: Data,
                        batch_idx: Tensor,
                        dataloader_idx: int) -> Dict:

        if dataloader_idx == 0:
            y = batch.y.squeeze()
            out = self.model(batch)
            loss = self.loss_criterion(out, y)
            self.log("val/loss",
                     loss,
                     prog_bar=False,
                     batch_size=self.args.batch_size,
                     add_dataloader_idx=False)
            self.val_evaluator.update(out, y)
        else:
            if self.test_eval_still != 0:
                return {'loader_idx': dataloader_idx}
            # only do validation on test set when reaching the predefined epoch.
            else:
                y = batch.y.squeeze()
                out = self.model(batch)
                loss = self.loss_criterion(out, y)
                self.log("test/loss",
                         loss,
                         prog_bar=False,
                         batch_size=self.args.batch_size,
                         add_dataloader_idx=False)
                self.test_evaluator.update(out, y)
        return {'loss': loss, 'preds': out, 'target': y, 'loader_idx': dataloader_idx}

    def on_validation_epoch_end(self) -> None:
        self.log("val/metric",
                 self.val_evaluator.compute(),
                 prog_bar=True,
                 add_dataloader_idx=False)
        self.val_evaluator.reset()
        if self.test_eval_still == 0:
            self.log("test/metric",
                     self.test_evaluator.compute(),
                     prog_bar=True,
                     add_dataloader_idx=False)
            self.test_evaluator.reset()
            self.test_eval_still = self.args.test_eval_interval
        else:
            self.test_eval_still = self.test_eval_still - 1

    def set_test_eval_still(self):
        # set test validation interval to zero to performance test dataset validation.
        self.test_eval_still = 0

    def on_test_epoch_start(self) -> None:
        self.set_test_eval_still()

    def test_step(self,
                  batch: Data,
                  batch_idx: Tensor,
                  dataloader_idx: int) -> Dict:
        results = self.validation_step(batch, batch_idx, dataloader_idx)
        return results

    def on_test_epoch_end(self) -> None:
        self.on_validation_epoch_end()




