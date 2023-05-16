"""
Utils file for model construction.
"""

from copy import deepcopy as c
from typing import Any

import torch.nn as nn
from torch_geometric.data import Data


def clones(module: nn.Module,
           N: int) -> nn.ModuleList:
    r"""Layer clone function, used for concise code writing. If input is None, simply return None.
    Args:
        module (nn.Module): Module want to clone.
        N (int): Clone times.
    """
    if module is None:
        return module
    else:
        return nn.ModuleList(c(module) for _ in range(N))


def get_pyg_attr(data: Data,
                 attr: str) -> Any:
    r"""Get attribute from PyG data. If not exist, return None instead.
    Args:
        data (torch_geometric.Data): PyG data object.
        attr (str): Attribute you want to get.
    """
    return getattr(data, attr, None)
