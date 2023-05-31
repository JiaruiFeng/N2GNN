"""
Utils file for training.
"""

import argparse
import os
import shutil
import time
import torch
import data_utils
import yaml
from torch_geometric.data import Data, Dataset
from sklearn.model_selection import StratifiedKFold
from typing import Callable, Tuple


def args_setup():
    r"""Setup argparser.
    """
    parser = argparse.ArgumentParser(f'arguments for training and testing')
    # common args
    parser.add_argument('--save_dir', type=str, default='./save', help='Base directory for saving information.')
    parser.add_argument('--config_file', type=str, default=None,
                        help='Additional configuration file for different dataset and models.')
    parser.add_argument('--seed', type=int, default=234, help='Random seed for reproducibility.')

    #training args
    parser.add_argument('--drop_prob', type=float, default=0.0,
                        help='Probability of zeroing an activation in dropout models.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size per GPU.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate.')
    parser.add_argument('--l2_wd', type=float, default=0., help='L2 weight decay.')
    parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs.')
    parser.add_argument('--test_eval_interval', type=int, default=10,
                        help='Interval between validation on test dataset.')
    parser.add_argument('--factor', type=float, default=0.5,
                        help='Factor in the ReduceLROnPlateau learning rate scheduler.')
    parser.add_argument('--patience', type=int, default=20,
                        help='Patience in the ReduceLROnPlateau learning rate scheduler.')
    parser.add_argument("--offline", action="store_true", help="If true, save the wandb log offline. "
                                                               "Mainly use for debug.")

    # data args
    parser.add_argument('--policy', default="dense_ego", choices=("dense_ego",
                                                                  "dense_noego",
                                                                  "sparse_ego",
                                                                  "sparse_noego"),
                        help="Policy of data generation in N2GNN. If dense, keep tuple that don't have any aggregation."
                             "if ego, further restrict all tuple mask have distance less than num_hops.")
    parser.add_argument('--message_pool', default="plain", choices=("plain", "hierarchical"),
                        help="message pooling way in N2GNN, if set to plain, pooling all edges together. If set to"
                             "hierarchical, compute index during preprocessing for hierarchical pooling, must be used"
                             "with corresponding gnn convolutional layer.")
    parser.add_argument('--reprocess', action="store_true", help='Whether to reprocess the dataset')

    # model args
    parser.add_argument('--gnn_name', type=str, default="GINEM", choices=("GINEC", "GINEM"),
                        help='Name of base gnn encoder.')
    parser.add_argument('--model_name', type=str, default="N2GNN+",
                        choices=("N2GNN+", "N2GNN"), help='Name of GNN model.')
    parser.add_argument('--tuple_size', type=int, default=5, help="Length of tuple in tuple aggregation.")
    parser.add_argument('--num_hops', type=int, default=3, help="Number of hop in ego-net selection.")
    parser.add_argument("--hidden_channels", type=int, default=96, help="Hidden size of the model.")
    parser.add_argument('--wo_node_feature', action='store_true',
                        help='If true, remove node feature from model.')
    parser.add_argument('--wo_edge_feature', action='store_true',
                        help='If true, remove edge feature from model.')
    parser.add_argument("--edge_dim", type=int, default=0, help="Number of edge type.")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of layer for GNN.")
    parser.add_argument("--JK", type=str, default="last",
                        choices=("sum", "max", "mean", "attention", "last", "concat"), help="Jumping knowledge method.")
    parser.add_argument("--residual", action="store_true", help="If ture, use residual connection between each layer.")
    parser.add_argument("--eps", type=float, default=0., help="Initial epsilon in GIN.")
    parser.add_argument("--train_eps", action="store_true", help="If true, the epsilon is trainable.")
    parser.add_argument("--pooling_method", type=str, default="mean", choices=("mean", "sum", "attention"),
                        help="Pooling method in graph level tasks.")
    parser.add_argument('--norm_type', type=str, default="Batch",
                        choices=("Batch", "Layer", "Instance", "GraphSize", "Pair", "None"),
                        help="Normalization method in model.")
    parser.add_argument('--add_rd', action="store_true", help="If true, additionally add resistance distance into model.")
    return parser


def get_exp_name(args: argparse.ArgumentParser, add_task=True) -> str:
    """Get experiment name.
    Args:
        args (ArgumentParser): Arguments dict from argparser.
    """

    arg_list = []
    if "task" in args and add_task:
        arg_list = [str(args.task)]
    arg_list.extend([args.dataset_name,
                     args.gnn_name,
                     args.model_name,
                     str(args.num_layers),
                     str(args.hidden_channels)
                     ])

    arg_list.extend([args.policy, str(args.num_hops)])

    if args.residual:
        arg_list.append("residual")
    if args.add_rd:
        arg_list.append("rd")

    exp_name = "_".join(arg_list)
    return exp_name + f"-{time.strftime('%Y%m%d%H%M%S')}"


def update_args(args: argparse.ArgumentParser, add_task=True) -> argparse.ArgumentParser:
    r"""Update argparser given config file.
    Args:
        args (ArgumentParser): Arguments dict from argparser.
    """

    if args.config_file is not None:
        with open(args.config_file) as f:
            cfg = yaml.safe_load(f)
        for key, value in cfg.items():
            if isinstance(value, list):
                for v in value:
                    getattr(args, key, []).append(v)
            else:
                setattr(args, key, value)

    args.exp_name = get_exp_name(args, add_task)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.message_pool == "plain":
        assert args.gnn_name in ["GINEC", "GINEM"]
    elif args.message_pool == "hierarchical":
        assert args.gnn_name in ["GINECH", "GINEMH"]

    return args


def data_setup(args: argparse.ArgumentParser) -> Tuple[str, Callable, list]:
    r"""Setup data for experiment.
    Args:
        args (ArgumentParser): Arguments dict from argparser.
    """
    update_style = args.model_name[:5]
    path_arg_list = [f"data/{args.dataset_name}"]
    path_arg_list.extend([update_style, str(args.num_hops), args.policy, args.message_pool])

    sparse = False
    ego_net = True
    hierarchical = False
    add_rd = False
    if args.policy == "sparse_ego":
        sparse = True
    elif args.policy == "sparse_noego":
        sparse = True
        ego_net = False
    elif args.policy == "dense_noego":
        ego_net = False
    if args.message_pool == "hierarchical":
        hierarchical = True
    if args.add_rd:
        add_rd = True

    pre_transform = data_utils.get_data_transform(update_style,
                                                  args.num_hops,
                                                  sparse,
                                                  ego_net,
                                                  hierarchical,
                                                  add_rd)

    follow_batch = []
    path = "_".join(path_arg_list)
    if os.path.exists(path + "/processed") and args.reprocess:
        shutil.rmtree(path + "/processed")

    return path, pre_transform, follow_batch


class PostTransform(object):
    r"""Post transformation of dataset.
    Args:
        wo_node_feature (bool): If true, remove path encoding from model.
        wo_edge_feature (bool): If true, remove edge feature from model.
        task (int): Specify the task in dataset if it has multiple targets.
    """
    def __init__(self,
                 wo_node_feature: bool,
                 wo_edge_feature: bool,
                 task: int = None):
        self.wo_node_feature = wo_node_feature
        self.wo_edge_feature = wo_edge_feature
        self.task = task

    def __call__(self,
                 data: Data) -> Data:
        if "x" not in data:
            data.x = torch.zeros([data.num_nodes, 1]).long()

        if self.wo_edge_feature:
            data.edge_attr = None
        if self.wo_node_feature:
            data.x = torch.zeros_like(data.x)
        if self.task is not None:
            data.y = data.y[:, self.task]
        return data


def k_fold(dataset: Dataset,
           folds: int,
           seed: int) -> Tuple[list, list, list]:
    r"""Dataset split for K-fold cross-validation.
    Args:
        dataset (Dataset): The dataset to be split.
        folds (int): Number of folds.
        seed (int): Random seed.
    """
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y[dataset.indices()]):
        test_indices.append(torch.from_numpy(idx).long())

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset)).long()
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero().view(-1))

    return train_indices, test_indices, val_indices


def get_seed(seed=234) -> int:
    r"""Return random seed based on current time.
    Args:
        seed (int): base seed.
    """
    t = int(time.time() * 1000.0)
    seed = seed + ((t & 0xff000000) >> 24) + ((t & 0x00ff0000) >> 8) + ((t & 0x0000ff00) << 8) + ((t & 0x000000ff) << 24)
    return seed

