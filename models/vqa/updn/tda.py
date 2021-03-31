# --------------------------------------------------------
# OpenVQA
# Written by Zhenwei Shao https://github.com/ParadoxZW
# based on the implementation in https://github.com/hengyuan-hu/bottom-up-attention-vqa
# ELU is chosen as the activation function in non-linear layers due to
# the experiment results that indicate ELU is better than ReLU in BUTD model.
# --------------------------------------------------------

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
import torch
import math


# ------------------------------
# ----- Weight Normal MLP ------
# ------------------------------

class MLP(nn.Module):
    """
    class for non-linear fully connect network
    """

    def __init__(self, dims, act='ELU', dropout_r=0.0):
        super(MLP, self).__init__()

        layers = []
        for i in range(len(dims) - 1):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if dropout_r > 0:
                layers.append(nn.Dropout(dropout_r))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if act != '':
                layers.append(getattr(nn, act)())

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


# ------------------------------
# ---Top Down Attention Map ----
# ------------------------------


class AttnMap(nn.Module):
    '''
    implementation of top down attention
    '''

    def __init__(self, img_feat_size, hidden_size, dropout_r):
        super(AttnMap, self).__init__()
        self.linear_q = weight_norm(
            nn.Linear(hidden_size, hidden_size), dim=None)
        self.linear_v = weight_norm(
            nn.Linear(img_feat_size, img_feat_size), dim=None)
        self.nonlinear = MLP(
            [img_feat_size + hidden_size, hidden_size], dropout_r=dropout_r)
        self.linear = weight_norm(nn.Linear(hidden_size, 1), dim=None)

    def forward(self, q, v):
        v = self.linear_v(v)
        q = self.linear_q(q)
        logits = self.logits(q, v)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, q, v):
        num_objs = v.size(1)
        q = q.unsqueeze(1).repeat(1, num_objs, 1)
        vq = torch.cat((v, q), 2)
        joint_repr = self.nonlinear(vq)
        logits = self.linear(joint_repr)
        return logits


# ------------------------------
# ---- Attended Joint Map ------
# ------------------------------


class TDA(nn.Module):
    def __init__(self, img_feat_size, hidden_size, dropout_r):
        super(TDA, self).__init__()

        self.v_att = AttnMap(img_feat_size, hidden_size, dropout_r)
        self.q_net = MLP([hidden_size, hidden_size])
        self.v_net = MLP([img_feat_size, hidden_size])

    def forward(self, q, v):
        att = self.v_att(q, v)
        atted_v = (att * v).sum(1)
        q_repr = self.q_net(q)
        v_repr = self.v_net(atted_v)
        joint_repr = q_repr * v_repr
        return joint_repr
