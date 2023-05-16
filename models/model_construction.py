"""
Model construction.
"""

from argparse import ArgumentParser
from models.GNNs import *
from models.gnn_convs import *
from models.input_encoder import *
from models.output_decoder import *


def make_gnn_layer(args: ArgumentParser) -> nn.Module:
    r"""Function to construct gnn layer.
    Args:
        args (ArgumentParser): Arguments dict from argparser.
    """

    HP = False
    if args.gnn_name in ["GINEC", "GINECH"]:
        conv_layer = GINETupleConcatConv
    elif args.gnn_name in ["GINEM", "GINEMH"]:
        conv_layer = GINETupleMultiplyConv
    else:
        raise ValueError("Not supported GNN type")

    if args.gnn_name.endswith("H"):
        HP = True

    gnn_layer = conv_layer(args.hidden_channels,
                           args.hidden_channels,
                           args.tuple_size,
                           initial_eps=args.eps,
                           train_eps=args.train_eps,
                           norm_type=args.norm_type,
                           HP=HP)

    return gnn_layer


def make_GNN(args: ArgumentParser,
             gnn_layer: nn.Module,
             edge_encoder: nn.Module,
             init_encoder: nn.Module) -> nn.Module:
    r"""Make GNN model given input parameters.
    Args:
        args (ArgumentParser): Arguments dict from argparser.
        gnn_layer (nn.Module): GNN layer.
        edge_encoder (nn.Module): Edge feature input encoder.
        init_encoder (nn.Module): Node feature initial encoder.
    """

    feature_encoders = []
    # Shortest path distance.
    feature_encoders.append(EmbeddingEncoder(args.num_hops + 2, args.hidden_channels))
    # Resitance distance.
    if args.add_rd:
        feature_encoders.append(RDEncoder(args.hidden_channels))

    add_root = False
    if args.model_name.endswith("+"):
        add_root = True

    gnn = N2GNN(num_layers=args.num_layers,
                gnn_layer=gnn_layer,
                init_encoder=init_encoder,
                edge_encoder=edge_encoder,
                feature_encoders=feature_encoders,
                norm_type=args.norm_type,
                residual=args.residual,
                initial_eps=args.eps,
                train_eps=args.train_eps,
                drop_prob=args.drop_prob,
                add_root=add_root)

    return gnn


def make_decoder(args: ArgumentParser,
                 embedding_model: nn.Module) -> nn.Module:
    r"""Make decoder layer for different dataset.
    Args:
        args (ArgumentParser): Arguments dict from argparser.
        embedding_model (nn.Module): Graph representation model, typically a gnn output node representation.
    """
    if args.dataset_name in ["ZINC", "ZINC_full", "StructureCounting", "QM9"]:
        model = GraphRegression(embedding_model, pooling_method=args.pooling_method)
    elif args.dataset_name in ["count_cycle", "count_graphlet"]:
        model = NodeRegression(embedding_model)
    else:
        model = GraphClassification(embedding_model, out_channels=args.out_channels, pooling_method=args.pooling_method)
    return model


def make_model(args: ArgumentParser,
               init_encoder: nn.Module = None,
               edge_encoder: nn.Module = None) -> nn.Module:
    r"""Make learning model given input arguments.
    Args:
        args (ArgumentParser): Arguments dict from argparser.
        init_encoder (nn.Module): Node feature initial encoder.
        edge_encoder (nn.Module): Edge feature encoder.
    """

    gnn_layer = make_gnn_layer(args)
    gnn = make_GNN(args, gnn_layer, edge_encoder, init_encoder)
    model = make_decoder(args, gnn)
    return model
