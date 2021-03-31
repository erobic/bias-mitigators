# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from models.vqa.make_mask import make_mask
from models.vqa.ops.fc import FC, MLP
from models.vqa.ops.layer_norm import LayerNorm
from models.vqa.mcan.mca import MCA_ED
# from models.vqa.mcan.adapter import Adapter
from models.vqa.vqa_adapter import VQAAdapter
# from openvqa.models.mcan.mca import MCA_ED
# from openvqa.models.mcan.adapter import Adapter

import torch.nn as nn
import torch.nn.functional as F
import torch


# ------------------------------
# ---- Flatten the sequence ----
# ------------------------------

class AttFlat(nn.Module):
    def __init__(self, hidden_size=1024, flat_mlp_size=512, flat_glimpses=1, dropout_r=0.1, flat_out_size=2048):
        super(AttFlat, self).__init__()
        self.flat_glimpses = flat_glimpses

        self.mlp = MLP(
            in_size=hidden_size,
            mid_size=flat_mlp_size,
            out_size=flat_glimpses,
            dropout_r=dropout_r,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            hidden_size * flat_glimpses,
            flat_out_size
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.flat_glimpses):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted


# -------------------------
# ---- Main MCAN Model ----
# -------------------------

class MCAN(nn.Module):
    def __init__(self, pretrained_emb,
                 token_size,
                 answer_size,
                 word_embed_size=300,
                 use_glove=True,
                 hidden_size=1024,
                 flat_out_size=2048,
                 flat_mlp_size=512,
                 flat_glimpses=1,
                 ff_size=4096,
                 dropout_r=0.1,
                 multi_head=8):
        super(MCAN, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=word_embed_size
        )

        # Loading the GloVe embedding weights
        if use_glove:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm = nn.LSTM(
            input_size=word_embed_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.adapter = VQAAdapter(hidden_size=hidden_size)

        self.backbone = MCA_ED(hidden_size, ff_size, dropout_r, multi_head)

        # Flatten to vector
        self.attflat_img = AttFlat(hidden_size,
                                   flat_mlp_size=flat_mlp_size,
                                   flat_glimpses=flat_glimpses,
                                   dropout_r=dropout_r,
                                   flat_out_size=flat_out_size
                                   )
        self.attflat_lang = AttFlat(hidden_size,
                                    flat_mlp_size=flat_mlp_size,
                                    flat_glimpses=flat_glimpses,
                                    dropout_r=dropout_r,
                                    flat_out_size=flat_out_size)

        # Classification layers
        self.proj_norm = LayerNorm(flat_out_size)
        self.proj = nn.Linear(flat_out_size, answer_size)

    def forward(self, frcn_feat, bbox_feat, ques_ix):
        # Pre-process Language Feature
        lang_feat_mask = make_mask(ques_ix.unsqueeze(2))
        lang_feat = self.embedding(ques_ix)
        lang_feat, _ = self.lstm(lang_feat)

        img_feat, img_feat_mask = self.adapter(frcn_feat, bbox_feat)

        # Backbone Framework
        lang_feat, img_feat = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask
        )

        # Flatten to vector
        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )

        img_feat = self.attflat_img(
            img_feat,
            img_feat_mask
        )

        # Classification layers
        proj_feat = lang_feat + img_feat
        proj_feat = self.proj_norm(proj_feat)
        proj_feat = self.proj(proj_feat)

        return {
            'question_features': lang_feat,
            'logits': proj_feat
        }

        # return proj_feat
