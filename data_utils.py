"""
Utils file for processing data.
"""

import numpy as np
import torch
from torch import Tensor, LongTensor
from torch_geometric.data import Data
import scipy.sparse as sparse
import networkx as nx
from models.utils import get_pyg_attr


def edge_list_to_sparse_adj(edge_list: np.ndarray,
                            num_nodes: int) -> sparse.coo_matrix:
    r"""Convert graph edge list to a sparse adjacency matrix.
    Args:
        edge_list (np.ndarray): Edge list of the graph.
        num_nodes (int): Number of nodes in the graph
    """
    coo = sparse.coo_matrix(([1 for _ in range(edge_list.shape[-1])], (edge_list[0, :], edge_list[1, :])),
                            shape=(num_nodes, num_nodes))
    return coo


def shortest_dist_sparse_mult(adj_mat: sparse.coo_matrix,
                              hop: int = 6,
                              source: int = None) -> np.ndarray:
    r"""Compute the shortest path distance given a graph adjacency matrix.
    Args:
        adj_mat (sparse.coo_matrix): Sparse graph adjacency matrix.
        hop (int): The maximum number of hop to consider when computing the shortest path distance.
        source (int): Source node for compute the shortest path distance.
                      If not specified, return the shortest path distance matrix.
    """
    if source is not None:
        neighbor_adj = adj_mat[source]
        ind = source
    else:
        neighbor_adj = adj_mat
        ind = np.arange(adj_mat.shape[0])
    neighbor_adj_set = [neighbor_adj]
    neighbor_dist = neighbor_adj.todense()
    for i in range(hop - 1):
        new_adj = neighbor_adj_set[i].dot(adj_mat)
        neighbor_adj_set.append(new_adj)
        update_ind = (new_adj.sign() - np.sign(neighbor_dist)) == 1
        r, c = update_ind.nonzero()
        neighbor_dist[r, c] = i + 2
    neighbor_dist[neighbor_dist < 1] = -9999
    neighbor_dist[np.arange(len(neighbor_dist)), ind] = 0
    return np.asarray(neighbor_dist)


def extract_spd_feature(data: Data,
                        num_hops: int) -> Data:
    r"""Extract the shortest path distance features given PyG data object.
    Args:
        data (Data): A PyG graph data.
        num_hops (int): Number of components to be kept.
    """
    num_nodes = data.num_nodes
    edge_list = data.edge_index.numpy()
    dist = shortest_dist_sparse_mult(edge_list_to_sparse_adj(edge_list, num_nodes), hop=num_hops)
    dist[dist < 0] = num_hops + 1
    data.spd = torch.from_numpy(dist).long()
    return data


def extract_rd_feature(data: Data) -> Data:
    r"""Extract the resistance distance feature given PyG data object.
    Args:
        data (Data): A PyG graph data.
    """
    num_nodes = data.num_nodes
    edge_index = data.edge_index
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    adj[edge_index[0, :], edge_index[1, :]] = 1.0

    # 2) connected_components
    g = nx.Graph(adj)
    g_components_list = [g.subgraph(c).copy() for c in nx.connected_components(g)]
    g_resistance_matrix = np.zeros((num_nodes, num_nodes)) - 1.0
    g_index = 0
    for item in g_components_list:
        cur_adj = nx.to_numpy_array(item)
        cur_num_nodes = cur_adj.shape[0]
        cur_res_dis = np.linalg.pinv(
            np.diag(cur_adj.sum(axis=-1)) - cur_adj + np.ones((cur_num_nodes, cur_num_nodes),
                                                              dtype=np.float32) / cur_num_nodes
        ).astype(np.float32)
        A = np.diag(cur_res_dis)[:, None]
        B = np.diag(cur_res_dis)[None, :]
        cur_res_dis = A + B - 2 * cur_res_dis
        g_resistance_matrix[g_index:g_index + cur_num_nodes, g_index:g_index + cur_num_nodes] = cur_res_dis
        g_index += cur_num_nodes
    g_cur_index = []
    for item in g_components_list:
        g_cur_index.extend(list(item.nodes))
    g_resistance_matrix = g_resistance_matrix[g_cur_index, :]
    g_resistance_matrix = g_resistance_matrix[:, g_cur_index]
    g_resistance_matrix[g_resistance_matrix == -1.0] = 512.0
    data.rd = torch.from_numpy(g_resistance_matrix)
    return data


class TupleData(Data):
    r"""Data abstract class for 2-tuple based data. rewrite __inc__ function to adapt different increment
        value for some keys.
    """
    def __inc__(self,
                key,
                value,
                *args,
                **kwargs):
        if key in ["tuple2second", "original_edge_index"]:
            return self.original_num_nodes
        elif key == "first2second":
            return self.num_first
        elif key == "second2tuple":
            return self.num_nodes
        elif key == "node2graph":
            return 1
        else:
            return super().__inc__(key, value, *args, **kwargs)


class FWL2t:
    r"""Generate data for (2, t)-FWL+. The data is generated by first compute the shortest path distance matrix.
    Then, the initial feature of x(u, v) is set as the node feature of v. The spd(u, v) is added as additional encoding
    to have the isomorphic type of (u, v). Notes that is not the original definition of isomorphic type, but it will not bring any
    expressive power from theoretical perspective. The resistance distance feature is also computed and saved in case of
    further use. The tuple edge will be generated based on different update formula.

    Args:
        num_hops (int): Number of hop in ego network.
        sparse (bool): If true, delete tuple (u, v) that not in any aggregation to save memory.
        ego_net (bool): If true, only tuple (u, v) with SPD less or equal to num_hops can aggregate or receiving information.
        hierarchical (bool): If true, add index for hierarchical pooling in message passing.
        add_rd (bool): If true, add resistance distance as additional augmented feature.
    """
    def __init__(self,
                 num_hops: int,
                 sparse: bool = False,
                 ego_net: bool = True,
                 hierarchical: bool = False,
                 add_rd: bool = False):
        super().__init__()
        self.num_hops = num_hops
        self.sparse = sparse
        self.ego_net = ego_net
        self.hierarchical = hierarchical
        self.add_rd = add_rd
        self.encoding_functions = [lambda x: extract_spd_feature(x, num_hops)]
        if self.add_rd:
            self.encoding_functions.append(extract_rd_feature)

    def generate_tuple_edges(self,
                             edge_index: LongTensor,
                             mask: Tensor,
                             num_nodes: int,
                             edge_attr: Tensor = None,
                             ego_net: bool = True,
                             hierarchical: bool = False):
        raise NotImplemented

    def __call__(self,
                 data: Data) -> Data:
        assert data.is_undirected()
        num_nodes = data.num_nodes
        edge_index = data.edge_index
        edge_attr = get_pyg_attr(data, "edge_attr")
        for f in self.encoding_functions:
            data = f(data)
        spd = data.spd.contiguous().view(-1)
        distance_mask = torch.zeros_like(spd).long()
        distance_mask[spd <= self.num_hops] = 1

        return_list = self.generate_tuple_edges(edge_index,
                                                distance_mask,
                                                num_nodes,
                                                edge_attr,
                                                self.ego_net,
                                                self.hierarchical)

        if len(return_list) == 2:
            edge_index, edge_attr = return_list
            first2second, second2tuple, num_first = None, None, None
        else:
            edge_index, edge_attr, first2second, second2tuple = return_list

        x = get_pyg_attr(data, "x")
        if x is not None:
            x = torch.cat([x for _ in range(num_nodes)], dim=0)
        z0 = spd
        z0[spd > self.num_hops] = self.num_hops + 1
        if self.add_rd:
            z1 = data.rd.contiguous().view(-1)
        else:
            z1 = None
        embedding_mask = torch.zeros_like(spd).long()
        embedding_mask[torch.unique(edge_index[0])] = 1
        root_index = torch.tensor([i * num_nodes + i for i in range(num_nodes)]).long()
        #prevent isolated node be ruled out in sparse version.
        embedding_mask[root_index] = 1
        #(u, v) | v in V(G)
        #tuple2first = torch.repeat_interleave(torch.arange(num_nodes), num_nodes)
        #(u, v) | u in V(G)
        tuple2second = torch.arange(num_nodes).repeat(num_nodes)
        node2graph = torch.tensor([0 for _ in range(num_nodes)]).long()

        if self.sparse:
            # reindexing
            keep_index = embedding_mask.bool()
            node_index_dict = dict(zip(torch.arange(embedding_mask.size(0))[keep_index].tolist(),
                                  torch.arange(torch.sum(embedding_mask == 1)).tolist()))
            edge_index = edge_index.apply_(node_index_dict.get)
            root_index = root_index.apply_(node_index_dict.get)
            #tuple2first = tuple2first[keep_index]
            tuple2second = tuple2second[keep_index]
            z0 = z0[keep_index]
            if z1 is not None:
                z1 = z1[keep_index]
            if x is not None:
                x = x[keep_index]

        if first2second is not None:
            unique_edge_index = torch.unique(first2second)
            num_first = unique_edge_index.size(0)
            edge_index_dict = dict(zip(unique_edge_index.tolist(), torch.arange(num_first).tolist()))
            first2second = first2second.apply_(edge_index_dict.get)
            if self.sparse:
                second2tuple = second2tuple.apply_(node_index_dict.get)


        return TupleData(x=x,
                         z0=z0,
                         z1=z1,
                         edge_index=edge_index,
                         edge_attr=edge_attr,
                         first2second=first2second,
                         second2tuple=second2tuple,
                         num_first=num_first,
                         #tuple2first=tuple2first,
                         tuple2second=tuple2second,
                         node2graph=node2graph,
                         y=data.y,
                         root_index=root_index,
                         original_num_nodes=num_nodes,
                         #original_edge_index=data.edge_index,
                         #original_edge_attr=data.edge_attr,
                         num_nodes=x.size(0))


def compute_edge_product(index: int,
                         count: Tensor,
                         edge_index: LongTensor,
                         edge_attr: Tensor = None):
    r"""Given an input edge list and node index, compute product edges for this node. Namely,
    Given :math:   u, \forall  \{(u, v) | v \in V(G)\}, compute :math: `(w_1, w_2) \in \mathcal{N}(v) \times \mathcal{N}(u).

    Args:
        index (int): The index of node to compute deep product edge.
        count (Tensor): A tensor to save the number of edge for each node.
        edge_index (LongTensor): Input edge list for a graph.
        edge_attr (Tensor): If provided, compute product edge attr.

    """
    num_edges = edge_index.size(-1)
    w1_edge_product = torch.repeat_interleave(edge_index, dim=-1, repeats=count[index])
    w2_edge_product = edge_index[:, edge_index[0] == index].repeat(1, num_edges)

    if edge_attr is not None:
        if len(edge_attr.size()) == 2:
            # Support OGBG datasets which have multiple dimensions of edge features.
            w1_edge_attr_product = torch.repeat_interleave(edge_attr, dim=0, repeats=count[index])
            w2_edge_attr_product = edge_attr[edge_index[0] == index].repeat(torch.sum(count), 1)
        else:
            w1_edge_attr_product = torch.repeat_interleave(edge_attr, dim=-1, repeats=count[index])
            w2_edge_attr_product = edge_attr[edge_index[0] == index].repeat(torch.sum(count))

    else:
        w1_edge_attr_product = None
        w2_edge_attr_product = None

    return w1_edge_product, w1_edge_attr_product, w2_edge_product, w2_edge_attr_product


def generate_22_tuple_edges(edge_index: LongTensor,
                            mask: Tensor,
                            num_nodes: int,
                            edge_attr: Tensor = None,
                            ego_net: bool = True,
                            hierarchical: bool = False):
    r"""Generate tuple edges for N^2-GNN.
    Args:
        edge_index (LongTensor): Input edge list for a graph.
        mask (Tensor): A mask to indicate whether a tuple (u, v) exist in at least one aggregation.
        num_nodes (int): Number of nodes in the graph.
        edge_attr (Tensor): If provided, compute product edge attr.
        ego_net (bool): If true, only tuple (u, v) with SPD less or equal to num_hops can aggregate or receiving information.
        hierarchical (bool): If true, add index for hierarchical pooling in message passing.
    """
    count = torch.zeros([num_nodes]).long()
    unique, unique_count = torch.unique(edge_index[0], return_counts=True)
    count[unique] = unique_count


    uv_edge_index_list = []
    uw1_edge_index_list = []
    uw2_edge_index_list = []
    w1v_edge_index_list = []
    w2v_edge_index_list = []
    w1w2_edge_index_list = []
    uw1_edge_attr_list = []
    w2v_edge_attr_list = []
    first_to_second_list = []
    second_to_tuple_list = []
    edge_offset = 0
    for i in range(num_nodes):
        w1_subgraph_edges, w1_edge_attr, w2_subgraph_edges, w2_edge_attr = compute_edge_product(i,
                                                                                                count,
                                                                                                edge_index,
                                                                                                edge_attr)

        offset = num_nodes * i
        if hierarchical:
            # compute index for hierarchical pooling
            edge_inc = torch.arange(count[i])
            edge_inc_list = [edge_inc.repeat(count[j]) + j * count[i] for j in range(num_nodes)]
            first_to_second = torch.hstack(edge_inc_list) + edge_offset
            second_to_tuple = torch.hstack([torch.zeros(count[i], dtype=torch.long) + j for j in range(num_nodes)]) + offset

        uv_edge_index = w1_subgraph_edges[0] + offset
        uw1_edge_index = w1_subgraph_edges[1] + offset
        uw2_edge_index = w2_subgraph_edges[1] + offset
        w1w1_edge_index = w1_subgraph_edges[1] * num_nodes
        w1w2_edge_index = w1w1_edge_index + w2_subgraph_edges[1]
        w2w2_edge_index = w2_subgraph_edges[1] * num_nodes
        w2v_edge_index = w2w2_edge_index + w1_subgraph_edges[0]
        w1v_edge_index = w1w1_edge_index + w1_subgraph_edges[0]
        if ego_net:
            keep_index = (mask[uv_edge_index] +
                          mask[uw1_edge_index] +
                          mask[uw2_edge_index] +
                          mask[w1v_edge_index] +
                          mask[w2v_edge_index] +
                          mask[w1w2_edge_index]) == 6

        else:
            keep_index = (mask[uw1_edge_index] +
                          mask[w2v_edge_index]) == 2

        uv_edge_index_list.append(uv_edge_index[keep_index])
        uw1_edge_index_list.append(uw1_edge_index[keep_index])
        uw2_edge_index_list.append(uw2_edge_index[keep_index])
        w1v_edge_index_list.append(w1v_edge_index[keep_index])
        w2v_edge_index_list.append(w2v_edge_index[keep_index])
        w1w2_edge_index_list.append(w1w2_edge_index[keep_index])

        if hierarchical:
            first_to_second_list.append(first_to_second[keep_index])
            second_to_tuple_list.append(second_to_tuple)
            edge_offset += num_nodes * count[i]

        if w1_edge_attr is not None:
            uw1_edge_attr_list.append(w1_edge_attr[keep_index])

        if w2_edge_attr is not None:
            w2v_edge_attr_list.append(w2_edge_attr[keep_index])

    uv_edge_index = torch.cat(uv_edge_index_list)
    uw2_edge_index = torch.cat(uw2_edge_index_list)
    w1v_edge_index = torch.cat(w1v_edge_index_list)
    uw1_edge_index = torch.cat(uw1_edge_index_list)
    w2v_edge_index = torch.cat(w2v_edge_index_list)
    w1w2_edge_index = torch.cat(w1w2_edge_index_list)

    edge_index = torch.cat([uv_edge_index.unsqueeze(0),
                            uw1_edge_index.unsqueeze(0),
                            w2v_edge_index.unsqueeze(0),
                            uw2_edge_index.unsqueeze(0),
                            w1v_edge_index.unsqueeze(0),
                            w1w2_edge_index.unsqueeze(0)], dim=0)

    if hierarchical:
        first2second = torch.hstack(first_to_second_list)
        second2tuple = torch.hstack(second_to_tuple_list)
        second2tuple = second2tuple[torch.unique(first2second)]

    if edge_attr is not None:
        uw1_edge_attr = torch.cat(uw1_edge_attr_list)
        w2v_edge_attr = torch.cat(w2v_edge_attr_list)
        edge_attr = torch.cat([uw1_edge_attr.unsqueeze(-1),
                               w2v_edge_attr.unsqueeze(-1)], dim=-1)

    else:
        edge_attr = None
    if hierarchical:
        return edge_index, edge_attr, first2second, second2tuple
    else:
        return edge_index, edge_attr


class N2FWL(FWL2t):
    r"""Generate data for N2FWL with update formula as:
    ..math::
        x(u, v)= \text{HASH}(x(u, v) \{\{x(u, w_1), x(u, w_2), x(w_1, v), x(w_2, v), x(w_1, w_2)| (w_1, w_2) \in
                \mathcal{N}_{k}(u) \cap \mathcal{N}_{1}(v) \times \mathcal{N}_{k}(v) \cap \mathcal{N}_{1}(u)\}\}))
    """

    def generate_tuple_edges(self,
                             edge_index: LongTensor,
                             mask: Tensor,
                             num_nodes: int,
                             edge_attr: Tensor = None,
                             ego_net: bool = True,
                             hierarchical: bool = False):
        return generate_22_tuple_edges(edge_index,
                                       mask,
                                       num_nodes,
                                       edge_attr,
                                       ego_net,
                                       hierarchical)


def get_data_transform(model_name: str,
                       num_hops: int,
                       sparse: bool = False,
                       ego_net: bool = True,
                       hierarchical: bool = False,
                       add_rd: bool = False
                       ):
    r"""Given model name, return the corresponding data transform function.
    Args:
        model_name (str): The name of the model.
        num_hops (int): Number of hop in ego network.
        sparse (bool): If true, delete tuple (u, v) that not in any aggregation to save memory.
        ego_net (bool): If true, only tuple (u, v) with SPD less or equal to num_hops can aggregate or receiving information.
        hierarchical (bool): If true, add index for hierarchical pooling in message passing.
        add_rd (bool): If true, add resistance distance as additional augmented feature.
    """
    if model_name == "N2GNN":
        return N2FWL(num_hops, sparse, ego_net, hierarchical, add_rd)
    else:
        raise NotImplemented

