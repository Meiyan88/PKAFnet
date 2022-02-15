import torch_geometric.nn as tnn
import torch.nn as nn
import torch
from torch_geometric.nn.models import DeepGCNLayer
from torch_geometric.nn import EdgeConv, NNConv, GraphConv, TransformerConv
from torch_geometric.nn.pool.edge_pool import EdgePooling
from torch_geometric.nn import TopKPooling, GCNConv,GatedGraphConv, SAGPooling
import torch.nn.functional as F

class ResGCN(nn.Module):
    def __init__(self, gcn_params=None, time=None):
        super(ResGCN, self).__init__()

        ### Encoding
        self.sage1 = tnn.SAGEConv(gcn_params['in_channels'], gcn_params['out_channels1'], normalize=True)
        self.sage2 = tnn.SAGEConv(gcn_params['out_channels1'], gcn_params['out_channels2'], normalize=True)
        self.sage3 = tnn.SAGEConv(gcn_params['out_channels2'], gcn_params['out_channels3'], normalize=True)
        self.sage4 = tnn.SAGEConv(gcn_params['out_channels3'], gcn_params['out_channels4'], normalize=True)

        self.layer1 = self.make_layer(
            conv=tnn.SAGEConv(gcn_params['out_channels1'], gcn_params['out_channels1'], normalize=True),
            norm=nn.BatchNorm1d(num_features=gcn_params['out_channels1']), act=nn.LeakyReLU(True),
            block='res+',
            time=time[0])
        self.layer2 = self.make_layer(
            conv=tnn.SAGEConv(gcn_params['out_channels2'], gcn_params['out_channels2'], normalize=True),
            norm=nn.BatchNorm1d(num_features=gcn_params['out_channels2']), act=nn.LeakyReLU(True),
            block='res+',
            time=time[1])
        self.layer3 = self.make_layer(
            conv=tnn.SAGEConv(gcn_params['out_channels3'], gcn_params['out_channels3'], normalize=True),
            norm=nn.BatchNorm1d(num_features=gcn_params['out_channels3']), act=nn.LeakyReLU(True),
            block='res+',
            time=time[2])
        self.layer4 = self.make_layer(
            conv=tnn.SAGEConv(gcn_params['out_channels4'], gcn_params['out_channels4'], normalize=True),
            norm=nn.BatchNorm1d(num_features=gcn_params['out_channels4']), act=nn.LeakyReLU(True),
            block='res+',
            time=time[3])

        self.tr1 = nn.Linear(gcn_params['out_channels4'], gcn_params['out_channels5'])
        self.drop = torch.nn.Dropout(p=gcn_params['dropout'])
        # self.c2 = nn.Linear(gcn_params['out_channels5'], gcn_params['cate_class'])
        ## Batch Normalization
        self.bano1 = nn.BatchNorm1d(num_features=gcn_params['out_channels1'])
        self.bano2 = nn.BatchNorm1d(num_features=gcn_params['out_channels2'])
        self.bano3 = nn.BatchNorm1d(num_features=gcn_params['out_channels3'])
        self.bano4 = nn.BatchNorm1d(num_features=gcn_params['out_channels4'])
        self.bano5 = nn.BatchNorm1d(num_features=gcn_params['out_channels5'])

        self.edge1 = EdgePooling(gcn_params['out_channels1'], edge_score_method=None, add_to_edge_score=0.5)
        self.edge2 = EdgePooling(gcn_params['out_channels2'], edge_score_method=None, add_to_edge_score=0.5)
        self.edge3 = EdgePooling(gcn_params['out_channels3'], edge_score_method=None, add_to_edge_score=0.5)

    def make_layer(self, conv, norm, act, block, time):
        layer = []
        for i in range(time):
            layer.append(DeepGCNLayer(conv=conv,
                                      norm=norm, act=act, block=block))
        return nn.ModuleList(layer)

    def encode(self, x, adj, batch):
        hidden1 = self.sage1(x, adj)
        hidden1 = self.bano1(hidden1)
        hidden1 = F.leaky_relu(hidden1)
        for i in range(len(self.layer1)):
            hidden1 = self.layer1[i](hidden1, adj)
        hidden1, edge_index, batch, _ = self.edge1(hidden1, adj, batch)

        ### 2
        hidden1 = self.sage2(hidden1, edge_index)
        hidden1 = self.bano2(hidden1)
        hidden1 = F.leaky_relu(hidden1)
        for i in range(len(self.layer2)):
            hidden1 = self.layer2[i](hidden1, edge_index)
        hidden1, edge_index, batch, _ = self.edge2(hidden1, edge_index, batch)

        ### 3
        hidden1 = self.sage3(hidden1, edge_index)
        hidden1 = self.bano3(hidden1)
        hidden1 = F.leaky_relu(hidden1)
        for i in range(len(self.layer3)):
            hidden1 = self.layer3[i](hidden1, edge_index)
        hidden1, edge_index, batch, _ = self.edge3(hidden1, edge_index, batch)

        ### 4
        hidden1 = self.sage4(hidden1, edge_index)
        hidden1 = self.bano4(hidden1)
        hidden1 = F.leaky_relu(hidden1)
        for i in range(len(self.layer4)):
            hidden1 = self.layer4[i](hidden1, edge_index)

        slim = tnn.global_add_pool(hidden1, batch)

        slim = self.tr1(slim)
        slim = self.bano5(slim)
        slim = F.leaky_relu(slim)

        return slim

    def forward(self, x, adj, batch):
        feature_GCN_x = self.encode(x, adj, batch)  ## mu, log sigma
        # x= self.c2(feature_GCN_x)
        return feature_GCN_x

class Graph_encoder(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels2, out_channels3, out_channels4, out_channels5, dropout):
        super(Graph_encoder, self).__init__()
        self.sage1 = tnn.SAGEConv(in_channels, out_channels1, normalize=True)
        self.sage2 = tnn.SAGEConv(out_channels1, out_channels2, normalize=True)
        self.sage3 = tnn.SAGEConv(out_channels2, out_channels3, normalize=True)
        self.sage4 = tnn.SAGEConv(out_channels3, out_channels4, normalize=True)
        self.tr1 = nn.Linear(out_channels4, out_channels5)
        # self.tr2 = nn.Linear(out_channels5, cate_class)

        self.drop = torch.nn.Dropout(p=dropout)

        ## Batch Normalization
        self.bano1 = nn.BatchNorm1d(num_features=out_channels1)
        self.bano2 = nn.BatchNorm1d(num_features=out_channels2)
        self.bano3 = nn.BatchNorm1d(num_features=out_channels3)
        self.bano4 = nn.BatchNorm1d(num_features=out_channels4)
        self.bano5 = nn.BatchNorm1d(num_features=out_channels5)

        self.edge1 = EdgePooling(out_channels1, edge_score_method=None, dropout=dropout, add_to_edge_score=0.5)
        self.edge2 = EdgePooling(out_channels2, edge_score_method=None, dropout=dropout, add_to_edge_score=0.5)
        self.edge3 = EdgePooling(out_channels3, edge_score_method=None, dropout=dropout, add_to_edge_score=0.5)
    def forward(self, x, adj, batch):
        hidden1 = self.sage1(x, adj)
        hidden1 = self.bano1(hidden1)
        hidden1 = F.leaky_relu(hidden1)
        hidden1 = self.drop(hidden1)

        hidden1, edge_index, batch, _ = self.edge1(hidden1, adj, batch)

        ### 2
        hidden1 = self.sage2(hidden1, edge_index)
        hidden1 = self.bano2(hidden1)
        hidden1 = F.leaky_relu(hidden1)
        hidden1 = self.drop(hidden1)

        hidden1, edge_index, batch, _ = self.edge2(hidden1, edge_index, batch)

        ### 3
        hidden1 = self.sage3(hidden1, edge_index)
        hidden1 = self.bano3(hidden1)
        hidden1 = F.leaky_relu(hidden1)
        hidden1 = self.drop(hidden1)

        hidden1, edge_index, batch, _ = self.edge3(hidden1, edge_index, batch)

        ### 4
        hidden1 = self.sage4(hidden1, edge_index)
        hidden1 = self.bano4(hidden1)
        hidden1 = F.leaky_relu(hidden1)
        hidden1 = self.drop(hidden1)

        slim = tnn.global_add_pool(hidden1, batch)

        ### 5
        slim = self.tr1(slim)
        slim = self.bano5(slim)
        slim = F.leaky_relu(slim)

        # slim = self.tr2(slim)
        # slim=F.leaky_relu(slim)
        return slim