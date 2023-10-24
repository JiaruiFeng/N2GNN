# N2GNN
This repository is the official implementation of the N2-GNN proposed in the [[NeurIPS23]Extending the Design Space of Graph Neural Networks by Rethinking Folklore Weisfeiler-Lehman](https://arxiv.org/pdf/2306.03266.pdf) (previous named "Towards Arbitrarily Expressive GNNs in $O(n^2)$ Space by Rethinking Folklore Weisfeiler-Lehman").

### News
In version 2.0, we:
1. Improve the implementation of N2GNN. The current implementation is more memory saving than the previous one. The detail and new results will be provided in the camera-ready version of the paper. Stay tuned!
2. Add experiment for [BREC dataset](https://github.com/GraphPKU/BREC).
3. Minor bugs fixing and comment polishing. 

### Requirements
```
python=3.8
torch=2.0.1
PyG=2.3.1
pytorch_lightning=2.0.9
wandb=0.13.11
```
We also provide a [docker environment](https://hub.docker.com/repository/docker/wfrain/gracker/general) and corresponding [docker file](https://github.com/JiaruiFeng/python_docker) if users prefer a docker style setup. Before run the experiments, you may need to set up [wandb](https://docs.wandb.ai/quickstart#1.-set-up-wandb). 

### Usage
The result for all experiments will be uploaded to Wandb server with the project named as: 
`[task]_<dataset_name>_<GNN_layer>_<GNN_model>_<num_layer>_<hidden_channels>_<data_type>_<num_hop>_[rd]`.\
`task`: The number index of the task if dataset has multiple tasks (QM9; substructure counting).\
`GNN_layer`: Name of GNN layer. See detail in `models/gnn_conv.py`.\
`GNN_model`: Name of GNN model. See detail in `models/GNNs.py`.\
`num_layer`: Number of layer in GNN model.\
`hidden_channels`: Number of hidden channels for each layer.\
`data_type`: Data preprocessing type. See detail in `data_utils.py`.\
`num_hop`: Number of hop for overlapping subgraph.\
`rd`: Name ends with rd means model add resistance distance as additional feature. 

### Reproducibility
#### ZINC-Subset and ZINC-Full
For ZINC-Subset:
```
python train_zinc.py --config_file=configs/zinc.yaml
```
For ZINC-Full:
```
python train_zinc.py --config_file=configs/zinc_full.yaml
```

### Counting substructure
For `cycle counting`, set `data=count_cycle` and `task=0, 1, 2, 3` for 3-cycles, 4-cycles, 5-cycles, and 6-cycles, respectively.\
For `substructure counting`, set `data=count_graphlet` and `task=0, 1, 2, 3, 4` for tailed-triangles, chordal cycles, 4-cliques, 4-paths, and triangle-rectangle, respectively.
Additionally, set the number of hop `h` for different tasks. You can find exact setting in the paper.
```
python train_count.py --config_file=configs/count.yaml --task=$task --dataset_name=$data --num_hops=$h
```

### QM9
Run a single target:
```
python train_qm9.py --config_file=configs/qm9.yaml --task=7
```
Run all targets:
```
python train_qm9.py --config_file=configs/qm9.yaml --search
```

### SR25
```
python train_sr25.py --config_file=configs/sr25.yaml
```

### CSL
```
python train_CSL.py --config_file=configs/csl.yaml
```

### EXP
```
python train_EXP.py --config_file=configs/exp.yaml
```
### BREC
To run the experiment on BREC dataset, you need first download `brec_v3_no4v_60cfi.npy` from [BREC repository](https://github.com/GraphPKU/BREC) and put it in the `data/BREC` directory. Then, run the following command:
```
python train_BREC.py --config_file=configs/BREC.yaml
```

### Citation
If you find this work useful, please kindly cite our paper:
```
@inproceedings{
Feng2023extending,
title={Extending the Design Space of Graph Neural Networks by Rethinking Folklore Weisfeiler-Lehman},
author={Jiarui Feng and Lecheng Kong and Hao Liu and Dacheng Tao and Fuhai Li and Muhan Zhang and Yixin Chen},
booktitle={Advances in Neural Information Processing Systems},
year={2023}
}
```

