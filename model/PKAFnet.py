# encoding:utf-8
import torch.nn as nn
import torch
import torch.nn.functional as F
from multi_head_layer import MultiHeadAttention
from model.CNNEncoder import Unet
from model.GCNEncoder import Graph_encoder, ResGCN

__author__ = "Haoran Lai"


class PKAFnet(Unet):
    """
    PKAFnet based on Haoran Lai
    A uent module and GCN module

    """

    def __init__(self, encoder_name='se_resnet50', GCNencoder_name='resnet18',
                 int_channel=1, classes=1, gcn_params=None, p=0.4):
        super(PKAFnet, self).__init__(
            int_channel=int_channel,
            classes=classes,
            encoder_name=encoder_name
        )

        """
        Args:

        encoder_name: the encoder of Unet , chose for  'se_resnet50', 'resnext50'
        GCNencoder_name: the encoder of ResGCN, chose for 'simple', 'resnet18', 'resnet34', where ''simple is a simple GCN.
        int_channel: the input channle
        classes: class number
        gcn_params: a dict of channel of GCN

        Example:
        gcn_params = {'in_channels': 31, 
                      'out_channels1': 32,
                      'out_channels2': 64,
                      'out_channels3': 128,
                      'out_channels4': 256,
                      'out_channels5': 512,
                      'dropout': 0.0}
        """

        self.GCNencoder_name = GCNencoder_name
        if self.GCNencoder_name == 'simple':
            self.encode = Graph_encoder(gcn_params['in_channels'], gcn_params['out_channels1'],
                                        gcn_params['out_channels2'],
                                        gcn_params['out_channels3'], gcn_params['out_channels4'],
                                        gcn_params['out_channels5'], gcn_params['dropout'])
        elif self.GCNencoder_name == 'resnet18':
            self.encode = ResGCN(gcn_params, time=[2, 2, 2, 2])
        elif self.GCNencoder_name == 'resnet34':
            self.encode = ResGCN(gcn_params, time=[3, 4, 6, 3])

        self.multi_head_attention = MultiHeadAttention(n_head=4, d_model=gcn_params['out_channels5'],
                                                       d_k=gcn_params['out_channels5'] // 4,
                                                       d_v=gcn_params['out_channels5'] // 4,
                                                       use_residual=False)
        self.c2 = nn.Linear(gcn_params['out_channels5'] * 4, gcn_params['cate_class'])

        self.avg = nn.AdaptiveAvgPool3d(1)
        self.wh1 = nn.Linear(self.ecoder_channels[-1], gcn_params['out_channels5'])
        self.bnwh1 = nn.BatchNorm1d(gcn_params['out_channels5'])
        self.drop = nn.Dropout(p=p)

    def forward(self, data1, data2, data3, img):
        """
        :param data1:  graph input alone x axes
        :param data2:  graph input alone y axes
        :param data3:  graph input alone z axes
        :param img:    image input
        :return:
        mask: segmentation prediction
        result: class predic
        """

        features = self.encoder(img)
        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)
        if self.classification_head is not None:
            labels = self.classification_head(features[-1])

        feature_CNN = self.avg(features[-1]).view(features[-1].size(0), -1)
        feature_CNN = F.leaky_relu(self.bnwh1(self.wh1(feature_CNN)))

        x, adj, lengs = self.get_data_from_graph(data1)
        f1 = self.encode(x, adj, lengs)  ## mu, log sigma
        x, adj, lengs = self.get_data_from_graph(data2)
        f2 = self.encode(x, adj, lengs)  ## mu, log sigma
        x, adj, lengs = self.get_data_from_graph(data3)
        f3 = self.encode(x, adj, lengs)  ## mu, log sigma

        feature = torch.cat([feature_CNN.unsqueeze(1), f1.unsqueeze(1), f2.unsqueeze(1), f3.unsqueeze(1)], dim=1)
        feature, _ = self.multi_head_attention(feature, feature, feature)
        result = self.c2(self.drop(feature.view(feature.size(0), -1)))
        return masks, result

    def get_data_from_graph(self, data):
        gra = data.x.cuda()
        adj = data.edge_index.cuda()
        batch = data.batch.cuda()
        return gra, adj, batch





