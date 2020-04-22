#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from vegcn.models.utils import GraphConv, MeanAggregator


class GCN_V(nn.Module):
    def __init__(self, feature_dim, nhid, nclass, dropout=0):
        super(GCN_V, self).__init__()
        self.conv1 = GraphConv(feature_dim, nhid, MeanAggregator, dropout)

        self.nclass = nclass
        self.classifier = nn.Sequential(nn.Linear(nhid, nhid), nn.PReLU(nhid),
                                        nn.Linear(nhid, self.nclass))
        self.loss = torch.nn.MSELoss()

    def forward(self, data, output_feat=False, return_loss=False):
        assert not output_feat or not return_loss
        x, adj = data[0], data[1]
        x = self.conv1(x, adj)
        pred = self.classifier(x).view(-1)

        if output_feat:
            return pred, x

        if return_loss:
            label = data[2]
            loss = self.loss(pred, label)
            return pred, loss

        return pred


def gcn_v(feature_dim, nhid, nclass=1, dropout=0., **kwargs):
    model = GCN_V(feature_dim=feature_dim,
                  nhid=nhid,
                  nclass=nclass,
                  dropout=dropout)
    return model
