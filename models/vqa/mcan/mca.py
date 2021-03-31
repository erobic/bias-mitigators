# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from models.vqa.ops.fc import FC, MLP
from models.vqa.ops.layer_norm import LayerNorm

import torch.nn as nn
import torch.nn.functional as F
import torch
import math


# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, hidden_size, dropout_r, multi_head=8):
        super(MHAtt, self).__init__()
        self.multi_head = multi_head
        self.hidden_size = hidden_size

        self.linear_v = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_merge = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout_r)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.multi_head,
            int(self.hidden_size / self.multi_head)
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.multi_head,
            int(self.hidden_size / self.multi_head)
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.multi_head,
            int(self.hidden_size / self.multi_head)
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.hidden_size
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, hidden_size, ff_size, dropout_r):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=hidden_size,
            mid_size=ff_size,
            out_size=hidden_size,
            dropout_r=dropout_r,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self, hidden_size, ff_size, dropout_r, multi_head):
        super(SA, self).__init__()

        self.mhatt = MHAtt(hidden_size, dropout_r, multi_head)
        self.ffn = FFN(hidden_size, ff_size, dropout_r)

        self.dropout1 = nn.Dropout(dropout_r)
        self.norm1 = LayerNorm(hidden_size)

        self.dropout2 = nn.Dropout(dropout_r)
        self.norm2 = LayerNorm(hidden_size)

    def forward(self, y, y_mask):
        y = self.norm1(y + self.dropout1(
            self.mhatt(y, y, y, y_mask)
        ))

        y = self.norm2(y + self.dropout2(
            self.ffn(y)
        ))

        return y


# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------

class SGA(nn.Module):
    def __init__(self, hidden_size, ff_size, dropout_r, multi_head):
        super(SGA, self).__init__()

        self.mhatt1 = MHAtt(hidden_size, dropout_r, multi_head)
        self.mhatt2 = MHAtt(hidden_size, dropout_r, multi_head)
        self.ffn = FFN(hidden_size, ff_size, dropout_r)

        self.dropout1 = nn.Dropout(dropout_r)
        self.norm1 = LayerNorm(hidden_size)

        self.dropout2 = nn.Dropout(dropout_r)
        self.norm2 = LayerNorm(hidden_size)

        self.dropout3 = nn.Dropout(dropout_r)
        self.norm3 = LayerNorm(hidden_size)

    def forward(self, x, y, x_mask, y_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt1(v=x, k=x, q=x, mask=x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(v=y, k=y, q=x, mask=y_mask)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x


# ------------------------------------------------
# ---- MAC Layers Cascaded by Encoder-Decoder ----
# ------------------------------------------------

class MCA_ED(nn.Module):
    def __init__(self, hidden_size, ff_size, dropout_r, multi_head, layer=6):
        super(MCA_ED, self).__init__()

        self.enc_list = nn.ModuleList([SA(hidden_size, ff_size, dropout_r, multi_head) for _ in range(layer)])
        self.dec_list = nn.ModuleList([SGA(hidden_size, ff_size, dropout_r, multi_head) for _ in range(layer)])

    def forward(self, y, x, y_mask, x_mask):
        # Get encoder last hidden vector
        for enc in self.enc_list:
            y = enc(y, y_mask)

        # Input encoder last hidden vector
        # And obtain decoder last hidden vectors
        for dec in self.dec_list:
            x = dec(x, y, x_mask, y_mask)

        return y, x
