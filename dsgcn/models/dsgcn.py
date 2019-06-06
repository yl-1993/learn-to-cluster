#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.nn.functional as F

__all__ = ['dsgcn']


class GraphConv(nn.Module):

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj, D=None):
        if x.dim() == 3:
            xw = torch.matmul(x, self.weight)
            output = torch.bmm(adj, xw)
        elif x.dim() == 2:
            xw = torch.mm(x, self.weight)
            output = torch.spmm(adj, xw)
        if D is not None:
            output = output * 1. / D
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, dropout=0.0):
        super(BasicBlock, self).__init__()
        self.gc = GraphConv(inplanes, planes)
        self.relu = nn.ReLU(inplace=True)
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        else:
            self.dropout = None

    def forward(self, x, adj, D=None):
        x = self.gc(x, adj, D)
        x = self.relu(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class GCN(nn.Module):
    """ input is (bs ,N, D), for featureless, D=1
        output is (bs ,2)
    """

    def __init__(self, block, planes, feature_dim, featureless, num_classes=1, dropout=0.0, reduce_method='max', stage='dev'):
        if featureless:
            self.inplanes = 1
        else:
            self.inplanes = feature_dim
        self.num_classes = num_classes
        self.reduce_method = reduce_method
        super(GCN, self).__init__()

        assert feature_dim > 0
        assert dropout >= 0 and dropout < 1
        self.layers = self._make_layer(block, planes, dropout)
        self.classifier = nn.Linear(self.inplanes, num_classes)
        if stage == 'dev':
            self.loss = torch.nn.MSELoss()
        elif stage == 'seg':
            self.loss = torch.nn.NLLLoss()
        else:
            raise KeyError('Unknown stage: {}'.format(stage))

    def _make_layer(self, block, planes, dropout=0.0):
        layers = nn.ModuleList([])
        for i, plane in enumerate(planes):
            layers.append(block(self.inplanes, plane, dropout))
            self.inplanes = plane
        return layers

    def extract(self, x, adj):
        bs = x.size(0)
        adj.detach_()
        D = adj.sum(dim=2, keepdim=True)
        D.detach_()
        for layer in self.layers:
            x = layer(x, adj, D)
        # use global op to reduce N
        # make sure isomorphic graphs output the same representation
        if self.reduce_method == 'sum':
            x = torch.sum(x, dim=1)
        elif self.reduce_method == 'mean':
            x = torch.mean(x, dim=1)
        elif self.reduce_method == 'max':
            x = torch.max(x, dim=1)[0]
        elif self.reduce_method == 'no_pool':
            pass # wo global pooling
        else:
            raise KeyError('Unkown reduce method', self.reduce_method)
        x = x.view(-1, self.inplanes)
        x = self.classifier(x)
        if self.reduce_method == 'no_pool':
            if self.num_classes > 1:
                x = x.view(bs, -1, self.num_classes)
                x = torch.transpose(x, 1, 2).contiguous()
                x = F.log_softmax(x, dim=1)
            else:
                x = x.view(bs, -1)
        return x

    def forward(self, data, return_loss=False):
        x = self.extract(data[0], data[1])
        if return_loss:
            loss = self.loss(x.view(-1), data[2])
            return x, loss
        else:
            return x


def dsgcn(feature_dim, hidden_dims=[], featureless=True, \
        reduce_method='max', dropout=0.5, num_classes=1):
    model = GCN(BasicBlock,
                planes=hidden_dims,
                feature_dim=feature_dim,
                featureless=featureless,
                reduce_method=reduce_method,
                dropout=dropout,
                num_classes=num_classes)
    return model
