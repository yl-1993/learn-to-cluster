#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class MeanAggregator(nn.Module):
    def __init__(self):
        super(MeanAggregator, self).__init__()

    def forward(self, features, A):
        if features.dim() == 2:
            x = torch.spmm(A, features)
        elif features.dim() == 3:
            x = torch.bmm(A, features)
        else:
            raise RuntimeError('the dimension of features should be 2 or 3')
        return x


class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, agg, dropout=0):
        super(GraphConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Parameter(torch.FloatTensor(in_dim * 2, out_dim))
        self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        init.xavier_uniform_(self.weight)
        init.constant_(self.bias, 0)
        self.agg = agg()
        self.dropout = dropout

    def forward(self, features, A):
        feat_dim = features.shape[-1]
        assert (feat_dim == self.in_dim)
        agg_feats = self.agg(features, A)
        cat_feats = torch.cat([features, agg_feats], dim=-1)
        if features.dim() == 2:
            op = 'nd,df->nf'
        elif features.dim() == 3:
            op = 'bnd,df->bnf'
        else:
            raise RuntimeError('the dimension of features should be 2 or 3')
        out = torch.einsum(op, (cat_feats, self.weight))
        out = F.relu(out + self.bias)
        if self.dropout > 0:
            out = F.dropout(out, self.dropout, training=self.training)
        return out
