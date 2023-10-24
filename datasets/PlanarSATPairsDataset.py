"""
EXP dataset.
"""

import os
import pickle
from typing import Callable, Optional, List

import torch
from torch_geometric.data import InMemoryDataset, Data


class PlanarSATPairsDataset(InMemoryDataset):
    r"""EXP dataset.
    Args:
        root (str): Root path for saving dataset.
        dataname (str, optional): Dataset name for loading.
        transform (Callable, optional): Data transformation function after saving.
        pre_transform (Callable, optional): Data transformation function before saving.
    """

    def __init__(self,
                 root: str,
                 dataname: Optional[str] = "EXP",
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.dataname = dataname
        self.processed = os.path.join(root, self.dataname, "processed")
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        name = 'raw'
        return os.path.join("data", self.dataname, name)

    @property
    def raw_file_names(self) -> List[str]:
        return ["GRAPHSAT.pkl"]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = pickle.load(open(self.raw_paths[0], "rb"))
        data_list = [Data(**g.__dict__) for g in data_list]
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
