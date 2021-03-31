# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

import numpy as np
import glob, json, torch, random
import torch.utils.data as Data
import torch.nn as nn
from datasets.vqa.feat_filter import feat_filter
import logging


class BaseVQADataset(Data.Dataset):
    def __init__(self, load_visual_feats=True):
        self.token_to_ix = None
        self.pretrained_emb = None
        self.ans_to_ix = None
        self.ix_to_ans = None

        self.data_size = None
        self.token_size = None
        self.ans_size = None
        self.load_visual_feats = load_visual_feats

    def load_ques_ans(self, idx):
        raise NotImplementedError()

    def load_img_feats(self, idx, iid):
        raise NotImplementedError()

    def __getitem__(self, idx):
        vqa = self.load_ques_ans(idx)
        if self.load_visual_feats:
            frcn_feat_iter, bbox_feat_iter = self.load_img_feats(idx, vqa['image_id'])
            vqa['frcn_feat'] = torch.from_numpy(frcn_feat_iter)
            vqa['bbox_feat'] = torch.from_numpy(bbox_feat_iter)
        vqa['dataset_ix'] = idx
        vqa['question_token_ixs'] = torch.from_numpy(vqa['question_token_ixs'])
        vqa['y'] = torch.from_numpy(vqa['ans_iter'])
        # Adding 'group_name' suffix
        vqa['answer_group_name'] = vqa['answer']
        return vqa
        # return {
        #     'frcn_feat': torch.from_numpy(frcn_feat_iter),
        #     'bbox_feat': torch.from_numpy(bbox_feat_iter),
        #     'question_id': q_details['question_id'],
        #     'ques_ix_iter': torch.from_numpy(q_details['ques_ix_iter']),
        #     'answer': q_details['ans'],
        #     'y': torch.from_numpy(q_details['ans_iter']),
        #     'dataset_ix': idx,
        #     'local_group_name': q_details['local_grp_name'],
        #     'group_name': q_details['group_name'],
        #     'group_ix': q_details['group_ix']
        # }

    def __len__(self):
        return self.data_size

    def shuffle_list(self, list):
        random.shuffle(list)
