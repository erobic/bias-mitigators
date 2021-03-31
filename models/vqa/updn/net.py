# --------------------------------------------------------
# OpenVQA
# Written by Zhenwei Shao https://github.com/ParadoxZW
# --------------------------------------------------------

from models.vqa.updn.tda import TDA
from models.vqa.vqa_adapter import VQAAdapter

import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
import torch


# -------------------------
# ---- Main BUTD Model ----
# -------------------------

class UpDn(nn.Module):
    def __init__(self,
                 pretrained_emb,
                 token_size,
                 answer_size,
                 img_feat_size=1024,
                 word_embed_size=300,
                 hidden_size=1024,
                 use_glove=True,
                 flat_out_size=2048,
                 dropout_r=0.2,
                 classifier_dropout_r=0.5
                 ):
        super(UpDn, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=word_embed_size
        )

        # Loading the GloVe embedding weights
        if use_glove:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.rnn = nn.LSTM(
            input_size=word_embed_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.adapter = VQAAdapter(hidden_size=hidden_size)

        self.backbone = TDA(img_feat_size=hidden_size,
                            hidden_size=hidden_size,
                            dropout_r=dropout_r
                            )

        # Classification layers
        self.fc1 = weight_norm(nn.Linear(hidden_size, flat_out_size), dim=None)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(classifier_dropout_r, inplace=True)
        self.classifier = weight_norm(nn.Linear(flat_out_size, answer_size), dim=None)
        # layers = [
        #     weight_norm(nn.Linear(hidden_size,
        #                           flat_out_size), dim=None),
        #     nn.ReLU(),
        #     nn.Dropout(classifier_dropout_r, inplace=True),
        #     weight_norm(nn.Linear(flat_out_size, answer_size), dim=None)
        # ]
        # self.classifier = nn.Sequential(*layers)

    def forward(self, frcn_feat, bbox_feat, question_token_ixs):
        # Pre-process Language Feature
        # lang_feat_mask = make_mask(ques_ix.unsqueeze(2))
        lang_feat = self.embedding(question_token_ixs)
        lang_feat, _ = self.rnn(lang_feat)

        img_feat, _ = self.adapter(frcn_feat, bbox_feat)

        # Backbone Framework
        joint_feat = self.backbone(
            lang_feat[:, -1],
            img_feat
        )

        # Classification layers
        # proj_feat = self.classifier(joint_feat)
        fc1 = self.fc1(joint_feat)
        clf_in = self.dropout(self.relu(fc1))
        logits = self.classifier(clf_in)

        return {
            'logits': logits,
            'before_logits': fc1,
            'question_features': lang_feat[:, -1],
            'visual_features': img_feat,
            'joint_features': joint_feat
        }


class UpDnNoDropout(UpDn):
    def __init__(self,
                 pretrained_emb, token_size, answer_size):
        super().__init__(pretrained_emb, token_size, answer_size, dropout_r=0, classifier_dropout_r=0)
