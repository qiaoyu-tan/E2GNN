
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, GCNConv, SAGEConv, GATConv, APPNP, SGConv  # noqa
from torch.nn import Linear


class GCN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation=F.relu, dropout=0.5, norm=0):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.act = activation
        self.norm = norm
        # input layer
        self.layers.append(GCNConv(in_feats, n_hidden, cached=True))
        if norm:
            self.bns = torch.nn.ModuleList()
            self.bns.append(torch.nn.BatchNorm1d(n_hidden))
        # hidden layers
        for i in range(n_layers - 2):
            self.layers.append(GCNConv(n_hidden, n_hidden, cached=True))
            if norm:
                self.bns.append(torch.nn.BatchNorm1d(n_hidden))
        # output layer
        self.layers.append(GCNConv(n_hidden, n_classes))
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i, layer in enumerate(self.layers[0:-1]):
            x = layer(x, edge_index)
            if self.norm:
                x = self.bns[i](x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, edge_index)
        return x

    def testforward(self, data):
        x, edge_index = data.x, data.edge_index_test
        for i, layer in enumerate(self.layers[0:-1]):
            x = layer(x, edge_index)
            if self.norm:
                x = self.bns[i](x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, edge_index)
        return x

    def encode(self, data):
        x, edge_index = data.x, data.edge_index
        for i, layer in enumerate(self.layers[0:-1]):
            x = layer(x, edge_index)
            if self.norm:
                x = self.bns[i](x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # x = self.layers[-1](x, edge_index)
        return x


class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation=F.relu, dropout=0.5, norm=0):
        super(SAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.act = activation
        self.norm = norm
        if norm:
            self.bns = torch.nn.ModuleList()
            self.bns.append(torch.nn.BatchNorm1d(n_hidden))
        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden))
        # hidden layers
        for i in range(n_layers - 2):
            self.layers.append(SAGEConv(n_hidden, n_hidden))
            if norm:
                self.bns.append(torch.nn.BatchNorm1d(n_hidden))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes))
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i, layer in enumerate(self.layers[0:-1]):
            x = layer(x, edge_index)
            if self.norm:
                x = self.bns[i](x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, edge_index)
        return x

    def testforward(self, data):
        x, edge_index = data.x, data.edge_index_test
        for i, layer in enumerate(self.layers[0:-1]):
            x = layer(x, edge_index)
            if self.norm:
                x = self.bns[i](x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, edge_index)
        return x

    def encode(self, data):
            x, edge_index = data.x, data.edge_index
            for i, layer in enumerate(self.layers[0:-1]):
                x = layer(x, edge_index)
                if self.norm:
                    x = self.bns[i](x)
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            # x = self.layers[-1](x, edge_index)
            return x


class GAT(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, heads=8, activation=F.relu, dropout=0.5):
        super(GAT, self).__init__()
        self.layers = nn.ModuleList()
        self.act = activation
        self.heads = 8
        atten_dim = int(n_hidden / self.heads)
        # input layer
        self.layers.append(GATConv(in_channels=in_feats, out_channels=atten_dim, heads=heads, dropout=dropout))
        # hidden layers
        for i in range(n_layers - 2):
            self.layers.append(GATConv(in_channels=n_hidden, out_channels=atten_dim, heads=heads, dropout=dropout))
        # output layer
        self.layers.append(GATConv(in_channels=n_hidden, out_channels=n_classes, heads=1, dropout=dropout))
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for layer in self.layers[0:-1]:
            x = layer(x, edge_index)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, edge_index)
        return x

    def testforward(self, data):
        x, edge_index = data.x, data.edge_index_test
        for layer in self.layers[0:-1]:
            x = layer(x, edge_index)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, edge_index)
        return x

    def encode(self, data):
            x, edge_index = data.x, data.edge_index
            for layer in self.layers[0:-1]:
                x = layer(x, edge_index)
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            # x = self.layers[-1](x, edge_index)
            return x


class APPNPM(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, K=10, alpha=0.2, activation=F.relu, dropout=0.5):
        super(APPNPM, self).__init__()
        self.layers = nn.ModuleList()
        self.act = activation
        # input layer
        self.layers.append(Linear(in_feats, n_hidden))
        # hidden layers
        for i in range(n_layers - 2):
            self.layers.append(Linear(n_hidden, n_hidden))
        # output layer
        self.layers.append(Linear(n_hidden, n_classes))
        self.dropout = dropout
        self.prop1 = APPNP(K, alpha)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        for layer in self.layers[0:-1]:
            x = layer(x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x)
        x = self.prop1(x, edge_index)
        return x

    def testforward(self, data):
        x, edge_index = data.x, data.edge_index_test
        x = F.dropout(x, p=self.dropout, training=self.training)
        for layer in self.layers[0:-1]:
            x = layer(x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x)
        x = self.prop1(x, edge_index)
        return x


class SGC(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, K=2, activation=F.relu, dropout=0.5):
        super(SGC, self).__init__()
        # self.layers = nn.ModuleList()
        self.act = activation
        # # input layer
        # self.layers.append(Linear(in_feats, n_hidden))
        # # hidden layers
        # for i in range(n_layers - 2):
        #     self.layers.append(Linear(n_hidden, n_hidden))
        # # output layer
        self.lin = Linear(n_hidden, n_classes)
        self.dropout = dropout
        self.conv = SGConv(in_feats, n_hidden, K=K, cached=True)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv(x, edge_index)
        x = self.act(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)
        return x

    def testforward(self, data):
        x, edge_index = data.x, data.edge_index_test
        x = self.conv(x, edge_index)
        x = self.act(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)
        return x


class search_net(torch.nn.Module):
    def __init__(self, device=None, in_dim=7, out_dim=5,dropout=0.): #out_dim=num of teachers
        super(search_net, self).__init__()
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.device = device
        self.dropout=dropout

        self.module = nn.Sequential(nn.Linear(in_features=in_dim*out_dim, out_features=in_dim),
                                    nn.Dropout(dropout),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(in_features=in_dim, out_features=out_dim))

    def forward(self,data):
        data_extend=[]
        for i in range(0,data.shape[0]):
            data_extend.append(data[i])
        data=torch.cat(data_extend,axis=1)
        atten=self.module(data)
        atten=torch.softmax(atten,dim=1)
        return atten