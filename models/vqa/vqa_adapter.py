# --------------------------------------------------------
# Adapted from OpenVQA
# Written by Zhenwei Shao https://github.com/ParadoxZW
# --------------------------------------------------------

import torch.nn as nn
import torch
from models.vqa.make_mask import make_mask


class VQAAdapter(nn.Module):
    def __init__(self,
                 frcn_feat_size=(100, 2048),
                 bbox_feat_size=(100, 5),
                 bbox_feat_emb_size=1024,
                 use_bbox_feats=True,
                 hidden_size=1024
                 ):
        super(VQAAdapter, self).__init__()
        self.frcn_feat_size = frcn_feat_size
        self.bbox_feat_size = bbox_feat_size
        self.use_bbox_feats = use_bbox_feats
        self.hidden_size = hidden_size
        in_size = frcn_feat_size
        if self.use_bbox_feats:
            self.bbox_linear = nn.Linear(5, bbox_feat_emb_size)
            in_size = frcn_feat_size[1] + bbox_feat_emb_size
        self.frcn_linear = nn.Linear(in_size, hidden_size)

    def forward(self, frcn_feat, bbox_feat):

        img_feat_mask = make_mask(frcn_feat)

        if self.use_bbox_feats:
            bbox_feat = self.bbox_linear(bbox_feat)
            frcn_feat = torch.cat((frcn_feat, bbox_feat), dim=-1)
        img_feat = self.frcn_linear(frcn_feat)

        return img_feat, img_feat_mask
