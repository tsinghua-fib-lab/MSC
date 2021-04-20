# -*- coding: utf-8 -*-
"""
@author: zgz
"""

import torch
import torch.nn.functional as F
from torch_scatter import scatter_max, scatter_mean

from layers import EdgeLearning, ELConv


class MSC(torch.nn.Module):
    def __init__(self, args):
        super(MSC, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.num_edge_features = args.num_edge_features
        self.n_hidden = args.n_hidden
        self.dropout_ratio = args.dropout_ratio
        self.num_communities = args.num_communities
        self.n_embedding = args.n_embedding
        self.num_nodes = args.num_nodes
        self.device = args.device

        self.edge_embedding1 = torch.nn.Linear(self.num_edge_features, self.n_embedding)
        self.edge_embedding2 = torch.nn.Linear(self.num_edge_features, self.n_embedding)

        self.embedding1 = torch.nn.Linear(8, self.n_embedding)
        self.embedding2 = torch.nn.Linear(12, self.n_embedding)
        self.embedding3 = torch.nn.Linear(2*self.n_embedding, 2*self.n_embedding)

        self.edge_learning1 = EdgeLearning(2*self.n_embedding, self.n_embedding, 2*self.n_embedding)
        self.edge_learning2 = EdgeLearning(2*self.n_embedding, self.n_embedding, 2*self.n_embedding)

        self.conv_inter1 = ELConv(2*self.n_embedding, self.n_hidden, 2*self.n_embedding)
        self.conv_intra1 = ELConv(2*self.n_embedding, self.n_hidden, 2*self.n_embedding)

        self.conv_inter2 = ELConv(self.n_hidden, self.n_hidden, 2*self.n_embedding)
        self.conv_intra2 = ELConv(self.n_hidden, self.n_hidden, 2*self.n_embedding)

        self.attn_mlp = torch.nn.Linear(2*self.n_hidden, 2*self.n_hidden)

        self.lin1 = torch.nn.Linear(4*self.n_hidden, 2*self.n_hidden)
        self.lin2 = torch.nn.Linear(2*self.n_hidden, 1)

    def community_pooling(self, x, community_for_pooling, pooling_edges):
        x_i = x[pooling_edges[0, :]]
        x_j = x[pooling_edges[1, :]]
        x_pooling = torch.cat((x_i, x_j), dim=-1)
        x_pooling = F.relu(self.attn_mlp(x_pooling))

        community_for_pooling = community_for_pooling.view(-1, 1).repeat(1, x.size()[1]*2)

        res1 = scatter_mean(x_pooling, community_for_pooling, dim=0, dim_size=self.num_communities)
        res2, _ = scatter_max(x_pooling, community_for_pooling, dim=0, dim_size=self.num_communities)

        return torch.cat([res1, res2], dim=1)

    def forward(self, data):
        x, edge_index, edge_attr, community = data.x, data.edge_index, data.edge_attr, data.community
        community_idx_for_pooling, pooling_edges = data.community_idx_for_pooling, data.pooling_edges
        adj_inter, adj_intra = data.adj_inter, data.adj_intra
        edge_attr_inter, edge_attr_intra = data.edge_attr_inter, data.edge_attr_intra

        edge_attr_inter = F.relu(self.edge_embedding1(edge_attr_inter))
        edge_attr_intra = F.relu(self.edge_embedding2(edge_attr_intra))

        x1 = F.relu(self.embedding1(x[:,:8]))
        x2 = F.relu(self.embedding2(x[:,8:]))
        x = torch.cat([x1, x2], dim=1)
        x = F.relu(self.embedding3(x))

        mask_inter = self.edge_learning1(x, adj_inter, edge_attr_inter)
        mask_intra = self.edge_learning2(x, adj_intra, edge_attr_intra)

        x_intra = x.clone()
        x = F.relu(self.conv_inter1(x, adj_inter, mask_inter))
        x += F.relu(self.conv_intra1(x_intra, adj_intra, mask_intra))
        x1 = self.community_pooling(x, community_idx_for_pooling, pooling_edges)

        x_intra = x.clone()
        x = F.relu(self.conv_inter2(x, adj_inter, mask_inter))
        x += F.relu(self.conv_intra2(x_intra, adj_intra, mask_intra))
        x2 = self.community_pooling(x, community_idx_for_pooling, pooling_edges)

        x = x1 + x2

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.lin2(x).squeeze()

        return x
