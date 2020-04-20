#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        if nclass == 1:
            self.loss = nn.MSELoss()
        elif nclass == 2:
            self.loss = nn.NLLLoss()
        else:
            raise ValueError('nclass should be 1 or 2')

    def forward(self, data, return_loss=False):
        x, adj = data[0], data[1]
        x = self.conv1(x, adj)
        x = self.conv2(x, adj)
        x = self.conv3(x, adj)
        x = self.conv4(x, adj)
        x = x.view(-1, x.shape[-1])
        pred = self.classifier(x)
        pred = F.log_softmax(pred, dim=-1)

        if return_loss:
            label = data[2].view(-1)
            loss = self.loss(pred, label)
            return pred, loss

        return pred


def gcn_e(feature_dim, nhid=512, nclass=1, dropout=0., **kwargs):
    model = GCN_E(feature_dim=feature_dim,
                  nhid=nhid,
                  nclass=nclass,
                  dropout=dropout)
    return model
