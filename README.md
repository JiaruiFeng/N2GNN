# N2GNN
Source code for N2GNN.

### Requirements
```
python=3.8
torch=1.11.0
PyG=2.1.0
pytorch_lightning=1.9.4
torchmetrics=0.11.3
wandb=0.13.11
```
Before run the experiments, you need to set up [wandb](https://docs.wandb.ai/quickstart#1.-set-up-wandb). Additionally, we notice that Pytorch_lightning released version 2.0 with some major updates, which is not consistent with our adopted version. Please make sure you install the correct version. We will upgrade to the latest version as soon as possible.  

### Usage
The result for all experiments will be uploaded to Wandb server with the project named as: 
`<task>_<dataset_name>_(GNN_layer)_<GNN_model>_<num_layer>_<hidden_channels>_<data_type>_<num_hop>_<rd>`.\
`task`: The number index of the task if dataset has multiple tasks (QM9, count).\
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
For ZINC-FUll:
```
python train_zinc.py --config_file=configs/zinc_full.yaml
```

### Counting substructure
For `cycle counting`, set `data=count_cycle` and `task=0, 1, 2, 3` for 3-cycles, 4-cycles, 5-cycles, and 6-cycles, respectively.\
For `substructure counting`, set `data=count_graphlet` and `task=0, 1, 2, 3, 4` for tailed-triangles, chordal cycles, 4-cliques, 4-paths, and triangle-rectangle respectively.
```
python train_count.py --config_file=configs/count.yaml --task=$task --dataset_name=$data
```

### QM9
Run a single target:
```
python train_qm9.py --config_file=configs/qm9.yaml --task=7
```
Search for all targets:
```
python train_qm9.py --config_file=configs/qm9.yaml --search
```

### SR25
```
python train_sr35.py --config_file=configs/sr25.yaml
```

### CSL
```
python train_CSL.py --config_file=configs/csl.yaml
```