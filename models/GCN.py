import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, GCNConv, SAGEConv  # noqa


class GCN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation=F.relu, dropout=0.6):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.act = activation
        # input layer
        self.layers.append(GCNConv(in_feats, n_hidden, cached=True))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GCNConv(n_hidden, n_hidden, cached=True))
        # output layer
        self.layers.append(GCNConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        for i, layer in enumerate(self.layers[0:-1]):
            if i != 0:
                x = self.act(x)
                x = self.dropout(x)
            x = layer(x, edge_index, edge_weight)
        x = self.layers[-1](x, edge_index, edge_weight)
        return x


class ogb_GCN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout, use_linear=False):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.use_linear = use_linear

        self.convs = nn.ModuleList()
        if use_linear:
            self.linear = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(n_layers):
            in_hidden = n_hidden if i > 0 else in_feats
            out_hidden = n_hidden if i < n_layers - 1 else n_classes
            bias = i == n_layers - 1

            self.convs.append(GraphConv(in_hidden, out_hidden, "both", bias=bias))
            if use_linear:
                self.linear.append(nn.Linear(in_hidden, out_hidden, bias=False))
            if i < n_layers - 1:
                self.bns.append(nn.BatchNorm1d(out_hidden))

        self.dropout0 = nn.Dropout(min(0.1, dropout))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, feat):
        h = feat
        h = self.dropout0(h)
        for i in range(self.n_layers):
            conv = self.convs[i](self.g, h)
            if self.use_linear:
                linear = self.linear[i](h)
                h = conv + linear
            else:
                h = conv
            if i < self.n_layers - 1:
                h = self.bns[i](h)
                h = self.activation(h)
                h = self.dropout(h)
        return h
