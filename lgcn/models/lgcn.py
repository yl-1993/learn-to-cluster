import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class MeanAggregator(nn.Module):
    def __init__(self):
        super(MeanAggregator, self).__init__()

    def forward(self, features, A):
        x = torch.bmm(A, features)
        return x


class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, agg):
        super(GraphConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Parameter(torch.FloatTensor(in_dim * 2, out_dim))
        self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        init.xavier_uniform_(self.weight)
        init.constant_(self.bias, 0)
        self.agg = agg()

    def forward(self, features, A):
        b, n, d = features.shape
        assert (d == self.in_dim)
        agg_feats = self.agg(features, A)
        cat_feats = torch.cat([features, agg_feats], dim=2)
        out = torch.einsum('bnd,df->bnf', (cat_feats, self.weight))
        out = F.relu(out + self.bias)
        return out


class lgcn(nn.Module):
    def __init__(self, feature_dim):
        super(lgcn, self).__init__()
        self.bn0 = nn.BatchNorm1d(feature_dim, affine=False)
        self.conv1 = GraphConv(feature_dim, 512, MeanAggregator)
        self.conv2 = GraphConv(512, 512, MeanAggregator)
        self.conv3 = GraphConv(512, 256, MeanAggregator)
        self.conv4 = GraphConv(256, 256, MeanAggregator)

        self.classifier = nn.Sequential(nn.Linear(256, 256), nn.PReLU(256),
                                        nn.Linear(256, 2))
        self.loss = nn.CrossEntropyLoss()

    def extract(self, x, A, one_hop_idxs):
        # data normalization l2 -> bn
        B, N, D = x.shape
        # xnorm = x.norm(2,2,keepdim=True) + 1e-8
        # xnorm = xnorm.expand_as(x)
        # x = x.div(xnorm)

        x = x.view(-1, D)
        x = self.bn0(x)
        x = x.view(B, N, D)

        x = self.conv1(x, A)
        x = self.conv2(x, A)
        x = self.conv3(x, A)
        x = self.conv4(x, A)
        k1 = one_hop_idxs.size(-1)
        dout = x.size(-1)
        edge_feat = torch.zeros(B, k1, dout).cuda()
        for b in range(B):
            edge_feat[b, :, :] = x[b, one_hop_idxs[b]]
        edge_feat = edge_feat.view(-1, dout)
        pred = self.classifier(edge_feat)

        # shape: (B*k1, 2)
        return pred

    def forward(self, data, return_loss=False):
        x, A, one_hop_idxs, labels = data
        x = self.extract(x, A, one_hop_idxs)
        if return_loss:
            loss = self.loss(x, labels.view(-1))
            return x, loss
        else:
            return x
