
"""
BREC dataset. Adapted from https://github.com/GraphPKU/BREC
paper link: https://arxiv.org/abs/2304.07702
"""
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils.convert import from_networkx
from torch_geometric.loader import DataLoader
import os
from tqdm import tqdm
from torch_geometric.data import Batch, Data
from typing import Optional, Callable

part_name = ["Basic", "Regular", "Extension", "CFI", "Distance_Regular"]
# part_dict: {graph generation type, range}
part_dict = {
    "Basic": (0, 60),
    "Regular": (60, 160),
    "Extension": (160, 260),
    "CFI": (260, 320),
    # "4-Vertex_Condition": (360, 380),
    # "Distance_Regular": (380, 400),
    "Distance_Regular": (320, 340),
}

def graph6_to_pyg(x: str) -> Data:
    return from_networkx(nx.from_graph6_bytes(x))

class BRECDataset(InMemoryDataset):
    r"""BREC dataset.
    Args:
        name (str, optional): name suffix in saving the processed dataset.
        root (str, optional): Root path for saving dataset.
        transform (Callable, optional): Data transformation function after saving.
        pre_transform (Callable, optional): Data transformation function before saving.
        pre_filter (Callable, optional): Data filtering function.
        test_part (list, optional): parts for testing the model. Index corresponding to part_dict
        segment (int, optional): segmentation size for saving the processed data to saving memory during the processing.

    """
    def __init__(
        self,
        name: Optional[str] = "no_param",
        root: Optional[str] = "Data",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        test_part: Optional[list] = list(range(5)),
        segment: Optional[int] = 340
    ):
        self.root = root
        self.name = name
        self.test_part = test_part
        assert 340 % segment == 0
        self.segment = segment
        self.num_each_segment = 340 // segment
        super().__init__(root, transform, pre_transform, pre_filter)

        data_collection = []
        for i in range(self.segment):
            sub_data = torch.load(self.processed_paths[i])
            data_collection.extend(Batch.to_data_list(sub_data))
        keep_data_collection = []
        for part in self.test_part:
            start, end = part_dict[part_name[part]]
            keep_data_collection.extend(data_collection[start * 128: end * 128])
        self.data, self.slices = self.collate(data_collection)

    @property
    def processed_dir(self):
        name = "processed"
        return os.path.join(self.root, self.name, name)

    @property
    def raw_file_names(self):
        return ["brec_v3_no4v_60cfi.npy"]

    @property
    def processed_file_names(self):
        return [f"brec_v3_no4v_60cfi_{i}.pt" for i in range(self.segment)]


    def process(self):

        data_list = np.load("data/BREC/brec_v3_no4v_60cfi.npy", allow_pickle=True)
        data_collection = []
        for i in range(self.segment):
            start = i * self.num_each_segment
            end = (i + 1) * self.num_each_segment
            data_sub_list = []
            for i in range(start * 128, end * 128):
                data_sub_list.append(graph6_to_pyg(data_list[i]))
            data_collection.append(data_sub_list)

        for i in range(self.segment):
            if os.path.exists(self.processed_paths[i]):
                continue
            print(f"Processing part {str(i+1)}...")
            data_sub_list = data_collection[i]
            if self.pre_filter is not None:
                data_sub_list = [data for data in data_sub_list if self.pre_filter(data)]

            if self.pre_transform is not None:
                data_sub_list = [self.pre_transform(data) for data in tqdm(data_sub_list)]

            sub_data = Batch.from_data_list(data_sub_list)
            torch.save(sub_data, self.processed_paths[i])


def main():
    dataset = BRECDataset()
    print(len(dataset))
    print(dataset.data)
    print(dataset._data_list)
    print(dataset.indices()[10])
    print(dataset[0])
    # print(dataset[0])
    # print(dataset[400])
    loader = DataLoader(dataset, batch_size=2)
    for x in loader:
        print(x[0])
        break
    # print(dataset[0:2][0])


if __name__ == "__main__":
    main()