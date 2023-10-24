"""
Graph substructure counting dataset.
"""

import os
from typing import Callable, Optional, List

import numpy as np
import scipy.io as sio
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.data import Data


class GraphCountDatasetI2(InMemoryDataset):
    r"""Graph substructure counting dataset. Adapted from I2GNN paper : https://arxiv.org/pdf/2210.13978.pdf.
        Target name:
            For count_cycle:
                0: 3-cycle
                1: 4-cycle
                2: 5-cycle
                3: 6-cycle
            For count_graphlet:
                0: Tailed triangle
                1: Chordal cycle
                2: 4-Clique
                3: 4-Path
                4: Triangle-rectangle

    Args:
        dataname (str, optional): Dataset name for loading, choose from (count_cycle, count_graphlet).
        root (str, optional): Root path for saving dataset.
        split (str, optional): Dataset split, choose from (train, val, test).
        transform (Callable, optional): Data transformation function after saving.
        pre_transform (Callable, optional): Data transformation function before saving.
    """

    def __init__(self,
                 dataname: Optional[str] = 'count_cycle',
                 root: Optional[str] = 'data',
                 split: Optional[str] = 'train',
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.root = root
        self.dataname = dataname
        self.transform = transform
        self.pre_transform = pre_transform
        self.raw = os.path.join(root, dataname)
        self.processed = os.path.join(root, dataname, "processed")
        super(GraphCountDatasetI2, self).__init__(root=root, transform=transform, pre_transform=pre_transform)
        split_id = 0 if split == 'train' else 1 if split == 'val' else 2
        self.data, self.slices = torch.load(self.processed_paths[split_id])
        self.y_dim = self.data.y.size(-1)

    @property
    def raw_dir(self) -> str:
        name = 'raw'
        return os.path.join("data", self.dataname, name)

    @property
    def processed_dir(self) -> str:
        return self.processed

    @property
    def raw_file_names(self) -> List[str]:
        names = ["data"]
        return ['{}.mat'.format(name) for name in names]

    @property
    def processed_file_names(self) -> List[str]:
        return ['data_tr.pt', 'data_val.pt', 'data_te.pt']

    def download(self):
        pass

    def adj2data(self, A: np.ndarray, y: np.ndarray) -> Data:
        # x: (n, d), A: (e, n, n)
        # begin, end = np.where(np.sum(A, axis=0) == 1.)
        begin, end = np.where(A == 1.)
        edge_index = torch.tensor(np.array([begin, end]))
        # edge_attr = torch.argmax(torch.tensor(A[:, begin, end].T), dim=-1)
        # y = torch.tensor(np.concatenate((y[1], y[-1])))
        # y = torch.tensor(y[-1])
        # y = y.view([1, len(y)])

        # sanity check
        # assert np.min(begin) == 0
        num_nodes = A.shape[0]
        if y.ndim == 1:
            y = y.reshape([1, -1])
        return Data(edge_index=edge_index, y=torch.tensor(y), num_nodes=torch.tensor([num_nodes]))

    @staticmethod
    def wrap2data(d: dict) -> Data:
        # x: (n, d), A: (e, n, n)
        x, A, y = d['x'], d['A'], d['y']
        x = torch.tensor(x)
        begin, end = np.where(np.sum(A, axis=0) == 1.)
        edge_index = torch.tensor(np.array([begin, end]))
        edge_attr = torch.argmax(torch.tensor(A[:, begin, end].T), dim=-1)
        y = torch.tensor(y[-1:])
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    def process(self):
        # process npy data into pyg.Data
        print('Processing data from ' + self.raw_dir + '...')
        raw_data = sio.loadmat(self.raw_paths[0])
        if raw_data['F'].shape[0] == 1:
            data_list_all = [[self.adj2data(raw_data['A'][0][i], raw_data['F'][0][i]) for i in idx]
                             for idx in [raw_data['train_idx'][0], raw_data['val_idx'][0], raw_data['test_idx'][0]]]
        else:
            data_list_all = [[self.adj2data(A, y) for A, y in zip(raw_data['A'][0][idx][0], raw_data['F'][idx][0])]
                             for idx in [raw_data['train_idx'], raw_data['val_idx'], raw_data['test_idx']]]
        for save_path, data_list in zip(self.processed_paths, data_list_all):
            print('pre-transforming for data at' + save_path)
            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]
            if self.pre_transform is not None:
                temp = []
                for i, data in enumerate(data_list):
                    if i % 100 == 0:
                        print('Pre-processing %d/%d' % (i, len(data_list)))
                    data.num_nodes = data.num_nodes.item()
                    temp.append(self.pre_transform(data))
                data_list = temp
                # data_list = [self.pre_transform(data) for data in data_list]
            data, slices = self.collate(data_list)
            torch.save((data, slices), save_path)
