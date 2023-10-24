"""
Pytorch lightning data module for PyG dataset.
"""

from typing import Tuple, Optional, List
from lightning.pytorch import LightningDataModule
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader


class PlPyGDataModule(LightningDataModule):
    r"""Pytorch lightning data module for PyG dataset.
    Args:
        train_dataset (Dataset): Train PyG dataset.
        val_dataset (Dataset): Validation PyG dataset.
        test_dataset (Dataset): Test PyG dataset.
        batch_size (int, optional): Batch size.
        num_workers (int, optional): Number of process for data loader.
        follow_batch (list, optional): A list of key that will create a corresponding batch key in data loader.
        drop_last (bool, optional): If true, drop the last batch during the training to avoid loss/metric inconsistence.
    """

    def __init__(self,
                 train_dataset: Dataset,
                 val_dataset: Dataset,
                 test_dataset: Dataset,
                 batch_size: Optional[int] = 32,
                 num_workers: Optional[int] = 0,
                 follow_batch: Optional[List[str]] = [],
                 drop_last: Optional[bool] = True):
        super(PlPyGDataModule, self).__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.follow_batch = follow_batch
        self.drop_last = drop_last

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True,
                          follow_batch=self.follow_batch,
                          drop_last=self.drop_last)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False,
                          follow_batch=self.follow_batch)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False,
                          follow_batch=self.follow_batch)


class PlPyGDataTestonValModule(PlPyGDataModule):
    r"""In validation mode, return both validation and test set for validation.
        Should use with PlGNNTestonValModule .
    """

    def val_dataloader(self) -> Tuple[DataLoader, DataLoader]:
        return (DataLoader(self.val_dataset,
                           batch_size=self.batch_size,
                           num_workers=self.num_workers,
                           shuffle=False,
                           follow_batch=self.follow_batch),
                DataLoader(self.test_dataset,
                           batch_size=self.batch_size,
                           num_workers=self.num_workers,
                           shuffle=False,
                           follow_batch=self.follow_batch))

    def test_dataloader(self) -> Tuple[DataLoader, DataLoader]:
        return self.val_dataloader()
