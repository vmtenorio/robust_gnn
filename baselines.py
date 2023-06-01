"""
Script containing different baselines that will be used to compared the proposed architecture.
"""

import torch.nn as nn
from dgl.nn import GATConv, GraphConv



# class BaselineModel:
#     def __init__(self, model):
#         self.model = model


#     def evaluate_clas(self, features, labels, mask):
#         self.model.eval()


class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, gat_params,
                 act=nn.ELU(), last_act=None):
        super(GAT, self).__init__()
        self.layer1 = GATConv(in_dim, hidden_dim, num_heads, **gat_params)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = GATConv(hidden_dim * num_heads, out_dim, 1, **gat_params)
        self.act = act
        self.last_act = last_act

    def forward(self, graph, h):
        h = self.layer1(graph, h)
        # concatenate
        h = h.flatten(1)
        h = self.act(h)
        h = self.layer2(graph, h)
        
        if self.last_act:
            return self.last_act(h.squeeze())
        return h.squeeze()


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, act=nn.ELU(), last_act=None, bias=True):
        super(GCN, self).__init__()
        self.layer1 = GraphConv(in_dim, hidden_dim, bias=bias)
        self.layer2 = GraphConv(hidden_dim, out_dim, bias=bias)
        self.act = act
        self.last_act = last_act

    def forward(self, graph, h):
        h = self.layer1(graph, h)
        h = self.act(h)
        h = self.layer2(graph, h)

        if self.last_act:
            return self.last_act(h)
        return h