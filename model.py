import torch.nn as nn
import torch.nn.functional as F
from util import *

class LFSR(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(LFSR, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, g, input):
        g.ndata['h'] = input
        g.update_all(gcn_message_func, gcn_reduce_func)
        h = g.ndata.pop('h')
        h = self.linear(h)
        h = h.view((1, 1, 169, 1024))
        h = F.relu(self.conv1(h))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = h.view((169, 1024))
        return h

class LFSRN(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(LFSRN, self).__init__()
        self.gcn = LFSR(in_feats, out_feats)

    def forward(self, g, inputs):
        h = self.gcn(g, inputs)
        h = F.relu(h)
        return h

class LFAR(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(LFAR, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, feature):
        g.ndata['h'] = feature.t()
        g.update_all(gcn_message_func, gcn_reduce_func)
        g.ndata['h'] = F.relu(self.linear(g.ndata['h'].t())).t()
        return g.ndata.pop('h')

class LFARN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(LFARN, self).__init__()
        self.layers = LFAR(in_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, out_dim)

    def forward(self, g, input):
        h = self.layers(g, input.t())
        return self.linear(h.t()).t()

class LFARN_Z(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(LFARN_Z, self).__init__()
        self.layers = LFAR(in_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, out_dim)
        # self.avgpool = nn.AvgPool3d()

    def forward(self, g, input):
        h = self.layers(g, input.t())
        z = self.linear(h.t()).t()
        return self.linear(z.t()).t()
