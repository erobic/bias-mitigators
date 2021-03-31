# --------------------------------------------------------
# OpenVQA
# Written by Zhenwei Shao https://github.com/ParadoxZW
# --------------------------------------------------------

from models.vqa.make_mask import make_mask
from models.vqa.ops.fc import FC, MLP
from models.vqa.ops.layer_norm import LayerNorm
from models.vqa.ban._ban import _BAN
from models.vqa.vqa_adapter import VQAAdapter

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
import torch


# -------------------------
# ---- Main BAN Model ----
# -------------------------

class BAN(nn.Module):
    def __init__(self, pretrained_emb, token_size, answer_size,
                 word_embed_size=300,
                 img_feat_size=1024,
                 hidden_size=1024,
                 k_times=3,
                 dropout_r=0.2,
                 classifier_dropout_r=0.5,
                 glimpse=8,
                 flat_out_size=2048,
                 use_glove=True):
        super(BAN, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=word_embed_size
        )

        # Loading the GloVe embedding weights
        if use_glove:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.rnn = nn.GRU(
            input_size=word_embed_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.adapter = VQAAdapter(hidden_size=hidden_size)

        self.backbone = _BAN(img_feat_size,
                             hidden_size,
                             k_times,
                             dropout_r,
                             classifier_dropout_r,
                             glimpse)

        # Classification layers
        layers = [
            weight_norm(nn.Linear(hidden_size, flat_out_size), dim=None),
            nn.ReLU(),
            nn.Dropout(classifier_dropout_r, inplace=True),
            weight_norm(nn.Linear(flat_out_size, answer_size), dim=None)
        ]
        self.classifier = nn.Sequential(*layers)

    def forward(self, frcn_feat, bbox_feat, ques_ix):
        # Pre-process Language Feature
        # lang_feat_mask = make_mask(ques_ix.unsqueeze(2))
        lang_feat = self.embedding(ques_ix)
        lang_feat, _ = self.rnn(lang_feat)
        img_feat, _ = self.adapter(frcn_feat, bbox_feat)

        # Backbone Framework
        lang_feat = self.backbone(
            lang_feat,
            img_feat
        )

        # Classification layers
        proj_feat = self.classifier(lang_feat.sum(1))
        return {
            'question_features': lang_feat,
            'logits': proj_feat
        }


class BANNoDropout(BAN):
    def __init__(self, pretrained_emb, token_size, answer_size):
        super().__init__(pretrained_emb, token_size, answer_size,
                         dropout_r=0, classifier_dropout_r=0)
