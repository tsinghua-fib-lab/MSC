# -*- coding: utf-8 -*-
"""
@author: zgz
"""

import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing

from inits import glorot, zeros


class EdgeLearning(torch.nn.Module):
    r"""
    learn mask vector of each edge
    Args:
        dim_node: node feature dimensions
        dim_edge: edge feature dimensions

    Output:
        edge mask
    """

    def __init__(self, dim_node, dim_edge, out_dim, negative_slope=0.2, **kwargs):
        super(EdgeLearning, self).__init__()
        self.dim_node = dim_node
        self.dim_edge = dim_edge
        self.out_dim = out_dim
        self.negative_slope = negative_slope

        self.fc1 = torch.nn.Linear(dim_node * 2 + dim_edge, dim_node * 2)
        self.fc2 = torch.nn.Linear(dim_node * 2, out_dim)

    def forward(self, x, edge_index, edge_attr):
        x_i = x[edge_index[0, :]]
        x_j = x[edge_index[1, :]]
        mask = torch.cat([x_i, x_j, edge_attr], dim=-1)
        mask = F.leaky_relu(self.fc1(mask))
        mask = self.fc2(mask)
        # mask = torch.sigmoid(mask)

        return mask

    def __repr__(self):
        return '{}(dim_node={}, dim_edge={}, out_dim={})'.format(self.__class__.__name__,
                                                                 self.dim_node,
                                                                 self.dim_edge,
                                                                 self.out_dim)


class ELConv(MessagePassing):
    r"""
    edge learning conv
    Args:
        in_channels: input node feature dimensions
        out_channels: output node feature dimensions
        edge_dim: edge feature dimensions

    """

    def __init__(self, in_channels, out_channels, edge_dim,
                 negative_slope=0.2, bias=None, **kwargs):
        super(ELConv, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.negative_slope = negative_slope

        self.weight = Parameter(torch.Tensor(
            self.in_channels, self.out_channels))
        self.edge_weight = Parameter(
            torch.Tensor(self.edge_dim, self.out_channels))
        self.bias = Parameter(torch.Tensor(self.out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.edge_weight)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        x = torch.matmul(x, self.weight)
        edge_attr = torch.matmul(edge_attr, self.edge_weight)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, edge_attr, x_j):
        return x_j * edge_attr

    def update(self, aggr_out, x):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias

        return aggr_out + x

    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__,
                                                            self.in_channels,
                                                            self.out_channels)
