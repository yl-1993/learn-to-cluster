#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn

from vegcn.models.utils import GraphConv, MeanAggregator


class GCN_E(nn.Module):
    def __init__(self, feature_dim, nhid, nclass, dropout=0):
        super(GCN_E, self).__init__()
        nhid_half = int(nhid / 2)
        self.conv1 = GraphConv(feature_dim, nhid, MeanAggregator, dropout)
        self.conv2 = GraphConv(nhid, nhid, MeanAggregator, dropout)
        self.conv3 = GraphConv(nhid, nhid_half, MeanAggregator, dropout)
        self.conv4 = GraphConv(nhid_half, nhid_half, MeanAggregator, dropout)

        self.nclass = nclass
        self.classifier = nn.Sequential(nn.Linear(nhid_half, nhid_half),
                                        nn.PReLU(nhid_half),
                                        nn.Linear(nhid_half, self.nclass))

    def forward(self, x, adj):
        x = self.conv1(x, adj)
        x = self.conv2(x, adj)
        x = self.conv3(x, adj)
        x = self.conv4(x, adj)
        pred = self.classifier(x)

        return pred


def gcn_e(feature_dim, nhid=512, nclass=1, dropout=0., **kwargs):
    model = GCN_E(feature_dim=feature_dim,
                  nhid=nhid,
                  nclass=nclass,
                  dropout=dropout)
    return model
