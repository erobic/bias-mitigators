# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------
import os
import numpy as np
import glob, json, re
from datasets.vqa.base_vqa_dataset import BaseVQADataset
from datasets.vqa.ans_punct import prep_ans
from utils.data_utils import dict_collate_fn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Subset
import logging
from option import ROOT

class GQADataset(BaseVQADataset):
    def __init__(self,
                 data_dir,
                 split,
                 filename,
                 use_glove=True,
                 frcn_feat_size=(100, 2048),
                 bbox_feat_size=(100, 5),
                 prepare_group_map=False,
                 group_name_to_ix=None,
                 group_ix_to_name=None,
                 group_by='default',
                 alpha=0.2,
                 load_visual_feats=True):
        super(GQADataset, self).__init__(load_visual_feats=load_visual_feats)
        self.split = split
        self.frcn_feat_size = frcn_feat_size
        self.bbox_feat_size = bbox_feat_size
        self.ques_file_path = os.path.join(data_dir, filename)
        if group_by is None:
            group_by = 'default'
        self.group_by = group_by
        self.alpha = alpha

        # --------------------------
        # ---- Raw data loading ----
        # --------------------------
        ques_dict_preread = {
            split: json.load(open(os.path.join(data_dir, filename)))
        }
        # ques_dict_preread = {
        #     'train': json.load(open(__C.RAW_PATH[__C.DATASET]['train'], 'r')),
        #     'val': json.load(open(__C.RAW_PATH[__C.DATASET]['val'], 'r')),
        #     'testdev': json.load(open(__C.RAW_PATH[__C.DATASET]['testdev'], 'r')),
        #     'test': json.load(open(__C.RAW_PATH[__C.DATASET]['test'], 'r')),
        # }

        # Loading all image paths
        frcn_feat_path_list = glob.glob(data_dir + '/preprocessed/objects' + '/*.npz')

        # Loading question word list
        # stat_ques_dict = {
        #     **ques_dict_preread['train'],
        #     **ques_dict_preread['val'],
        #     **ques_dict_preread['testdev'],
        #     **ques_dict_preread['test'],
        # }

        # Loading answer word list
        # stat_ans_dict = {
        #     **ques_dict_preread['train'],
        #     **ques_dict_preread['val'],
        #     **ques_dict_preread['testdev'],
        # }

        # Loading question and answer list
        self.ques_dict = {}
        self.ques_dict = ques_dict_preread[split]

        # Define run data size
        self.data_size = self.ques_dict.__len__()
        logging.getLogger().info(f"split:  {split}")
        logging.getLogger().info(f' ========== Dataset size: {self.data_size}')

        # ------------------------
        # ---- Data statistic ----
        # ------------------------

        # {image id} -> {image feature absolutely path}
        self.iid_to_frcn_feat_path = self.img_feat_path_load(frcn_feat_path_list)

        # Loading dict: question dict -> question list
        self.qid_list = list(self.ques_dict.keys())
        if prepare_group_map:
            self.prepare_group_map()
        else:
            self.group_name_to_ix = group_name_to_ix
            self.group_ix_to_name = group_ix_to_name

        # Tokenize
        self.token_to_ix, self.pretrained_emb, max_token = self.tokenize('datasets/vqa/dicts.json',
                                                                         use_glove)
        self.token_size = self.token_to_ix.__len__()
        logging.getLogger().info(f' ========== Question token vocab size: {self.token_size}')

        self.max_token = -1
        if self.max_token == -1:
            self.max_token = max_token
        logging.getLogger().info(f'Max token length: {max_token} Trimmed to: {self.max_token}')

        # Answers statistic
        self.ans_to_ix, self.ix_to_ans = self.ans_stat('datasets/vqa/dicts.json')
        self.ans_size = self.ans_to_ix.__len__()
        logging.getLogger().info(f' ========== Answer token vocab size: {self.ans_size}')

    def prepare_head_tail_groups(self):
        """
        We divide answers within each local group into head or tail groups.
        Note that this does not take entropy as described in: https://arxiv.org/pdf/2006.05121.pdf into account.
        This function is meant to divide training data into head and tail groups.
        For validation/testdev data, we need to use: https://github.com/gqaood/GQA-OODs
        :return:
        """
        # Gather stats
        local_grp_to_ans_to_freq = {}  # local group name -> answer -> number of instances

        for qid in self.qid_list:
            q_details = self.ques_dict[qid]
            local_grp = self.filter_none(q_details['groups']['local'])
            ans = self.filter_none(q_details['answer'])
            if local_grp not in local_grp_to_ans_to_freq:
                local_grp_to_ans_to_freq[local_grp] = {}
            if ans not in local_grp_to_ans_to_freq[local_grp]:
                local_grp_to_ans_to_freq[local_grp][ans] = 0
            local_grp_to_ans_to_freq[local_grp][ans] += 1

        # Divide into head and tail based on alpha
        local_group_ans_to_head_tail = {}
        for local_grp in local_grp_to_ans_to_freq:
            freq_sum, freq_n = 0, 0
            for ans in local_grp_to_ans_to_freq[local_grp]:
                freq_sum += local_grp_to_ans_to_freq[local_grp][ans]
                freq_n += 1
            mean_freq = freq_sum / freq_n

            for ans in local_grp_to_ans_to_freq[local_grp]:
                curr_freq = local_grp_to_ans_to_freq[local_grp][ans]
                if curr_freq <= (1 + self.alpha) * mean_freq:
                    group_name = 'tail'
                else:
                    group_name = 'head'
                local_group_ans_to_head_tail[local_grp + '/' + ans] = group_name
        return local_group_ans_to_head_tail

    def prepare_group_map(self):
        self.group_name_to_ix = {}
        self.group_ix_to_name = {}
        group_ix = 0

        for qid in self.qid_list:
            q_details = self.ques_dict[qid]
            group_name = self.get_group_name(q_details)
            if group_name not in self.group_name_to_ix:
                self.group_name_to_ix[group_name] = group_ix
                group_ix += 1
        self.num_groups = group_ix
        # logging.getLogger().info(f"group_name_to_ix {self.group_name_to_ix}")
        logging.getLogger().info(f"# groups: {group_ix}")

    def get_group_name(self, q_details):
        local_grp_name = self.filter_none(q_details['groups']['local'])
        global_grp_name = self.filter_none(q_details['groups']['global'])
        qtype_detailed = self.filter_none(q_details['types']['detailed'])
        answer = self.filter_none(q_details['answer'])
        if self.group_by == 'answer':
            return answer
        elif self.group_by == 'local_group_name':
            return local_grp_name
        elif self.group_by == 'global_group_name':
            return global_grp_name
        elif self.group_by == 'qtype_detailed':
            return qtype_detailed
        elif self.group_by == 'head_tail':
            if not hasattr(self, 'local_group_ans_to_head_tail'):
                self.local_group_ans_to_head_tail = self.prepare_head_tail_groups()
            return self.local_group_ans_to_head_tail[local_grp_name + '/' + answer]
        elif self.group_by == 'default':
            # default
            return f"{local_grp_name}/{answer}"

    def img_feat_path_load(self, path_list):
        iid_to_path = {}

        for ix, path in enumerate(path_list):
            iid = path.split('/')[-1].split('.')[0]
            iid_to_path[iid] = path

        return iid_to_path

    # def tokenize(self, stat_ques_dict, use_glove):
    #     token_to_ix = {
    #         'PAD': 0,
    #         'UNK': 1,
    #         'CLS': 2,
    #     }
    #
    #     spacy_tool = None
    #     pretrained_emb = []
    #     if use_glove:
    #         spacy_tool = en_vectors_web_lg.load()
    #         pretrained_emb.append(spacy_tool('PAD').vector)
    #         pretrained_emb.append(spacy_tool('UNK').vector)
    #         pretrained_emb.append(spacy_tool('CLS').vector)
    #
    #     max_token = 0
    #     for qid in stat_ques_dict:
    #         ques = stat_ques_dict[qid]['question']
    #         words = re.sub(
    #             r"([.,'!?\"()*#:;])",
    #             '',
    #             ques.lower()
    #         ).replace('-', ' ').replace('/', ' ').split()
    #
    #         if len(words) > max_token:
    #             max_token = len(words)
    #
    #         for word in words:
    #             if word not in token_to_ix:
    #                 token_to_ix[word] = len(token_to_ix)
    #                 if use_glove:
    #                     pretrained_emb.append(spacy_tool(word).vector)
    #
    #     pretrained_emb = np.array(pretrained_emb)
    #
    #     return token_to_ix, pretrained_emb, max_token
    #
    #
    # def ans_stat(self, stat_ans_dict):
    #     ans_to_ix = {}
    #     ix_to_ans = {}
    #
    #     for qid in stat_ans_dict:
    #         ans = stat_ans_dict[qid]['answer']
    #         ans = prep_ans(ans)
    #
    #         if ans not in ans_to_ix:
    #             ix_to_ans[ans_to_ix.__len__()] = ans
    #             ans_to_ix[ans] = ans_to_ix.__len__()
    #
    #     return ans_to_ix, ix_to_ans

    def tokenize(self, json_file, use_glove):
        import en_vectors_web_lg
        token_to_ix, max_token = json.load(open(json_file, 'r'))[2:]
        spacy_tool = None
        if use_glove:
            spacy_tool = en_vectors_web_lg.load()

        pretrained_emb = []
        for word in token_to_ix:
            if use_glove:
                pretrained_emb.append(spacy_tool(word).vector)
        pretrained_emb = np.array(pretrained_emb)

        return token_to_ix, pretrained_emb, max_token

    def ans_stat(self, json_file):
        ans_to_ix, ix_to_ans = json.load(open(json_file, 'r'))[:2]

        return ans_to_ix, ix_to_ans

    # ----------------------------------------------
    # ---- Real-Time Processing Implementations ----
    # ----------------------------------------------

    def filter_none(self, x):
        if x is None:
            return 'none'
        else:
            return x

    def load_ques_ans(self, idx):
        qid = self.qid_list[idx]
        iid = self.ques_dict[qid]['imageId']
        q_details = self.ques_dict[qid]
        ques = self.ques_dict[qid]['question']
        question_token_ixs = self.proc_ques(ques, self.token_to_ix, max_token=self.max_token)
        ans_iter = np.zeros(1)
        local_grp_name = self.filter_none(q_details['groups']['local'])
        global_grp_name = self.filter_none(q_details['groups']['global'])
        group_name = self.get_group_name(q_details)
        if group_name in self.group_name_to_ix:
            group_ix = self.group_name_to_ix[group_name]
        else:
            group_ix = 0
        ans = self.ques_dict[qid]['answer']
        ans_iter = self.proc_ans(ans, self.ans_to_ix)
        #
        # 'head_tail': group_name,

        ret_obj = {
            'question_id': qid,
            'image_id': iid,
            'question_token_ixs': question_token_ixs,
            'ans_iter': ans_iter,
            'local_group_name': local_grp_name,
            'global_group_name': global_grp_name,
            'structural_group_name': q_details['types']['structural'],
            'semantic_group_name': q_details['types']['semantic'],
            'detailed_group_name': q_details['types']['detailed'],
            'qtype_detailed': q_details['types']['detailed'],
            'answer_group_name': ans,
            'group_name': group_name,
            'group_ix': group_ix,
            'answer': ans
        }

        if self.group_by == 'head_tail':
            ret_obj['head_tail'] = group_name
        return ret_obj
        # return qid, ques_ix_iter, ans_iter, iid, local_grp_name, group_name, group_ix

    def load_img_feats(self, idx, iid):
        frcn_feat = np.load(self.iid_to_frcn_feat_path[iid])
        frcn_feat_iter = self.proc_img_feat(frcn_feat['x'],
                                            img_feat_pad_size=self.frcn_feat_size[0])

        bbox_feat_iter = self.proc_img_feat(
            self.proc_bbox_feat(
                frcn_feat['bbox'],
                (frcn_feat['height'], frcn_feat['width'])
            ),
            img_feat_pad_size=self.bbox_feat_size[0]
        )

        return frcn_feat_iter, bbox_feat_iter

    # ------------------------------------
    # ---- Real-Time Processing Utils ----
    # ------------------------------------

    def proc_img_feat(self, img_feat, img_feat_pad_size):
        if img_feat.shape[0] > img_feat_pad_size:
            img_feat = img_feat[:img_feat_pad_size]

        img_feat = np.pad(
            img_feat,
            ((0, img_feat_pad_size - img_feat.shape[0]), (0, 0)),
            mode='constant',
            constant_values=0
        )

        return img_feat

    def proc_bbox_feat(self, bbox, img_shape):
        bbox_feat = np.zeros((bbox.shape[0], 5), dtype=np.float32)

        bbox_feat[:, 0] = bbox[:, 0] / float(img_shape[1])
        bbox_feat[:, 1] = bbox[:, 1] / float(img_shape[0])
        bbox_feat[:, 2] = bbox[:, 2] / float(img_shape[1])
        bbox_feat[:, 3] = bbox[:, 3] / float(img_shape[0])
        bbox_feat[:, 4] = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1]) / float(img_shape[0] * img_shape[1])

        return bbox_feat

    def proc_ques(self, ques, token_to_ix, max_token):
        ques_ix = np.zeros(max_token, np.int64)

        words = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            ques.lower()
        ).replace('-', ' ').replace('/', ' ').split()

        for ix, word in enumerate(words):
            if word in token_to_ix:
                ques_ix[ix] = token_to_ix[word]
            else:
                ques_ix[ix] = token_to_ix['UNK']

            if ix + 1 == max_token:
                break

        return ques_ix

    def proc_ans(self, ans, ans_to_ix):
        ans_ix = np.zeros(1, np.int64)
        ans = prep_ans(ans)
        ans_ix[0] = ans_to_ix[ans]

        return ans_ix


def prepare_gqa_subsets(data_dir, subset_ratio):
    train_set = GQADataset(data_dir, filename='train_balanced_questions.json', split='train',
                           prepare_group_map=True)
    data_size = train_set.data_size
    data_ixs = np.arange(0, data_size)
    np.random.shuffle(data_ixs)
    subset_size = int(data_size * subset_ratio)
    subset_ixs = data_ixs[0:subset_size].tolist()
    logging.getLogger().info(f"Saving {len(subset_ixs)} items...")
    json.dump(subset_ixs, open(os.path.join(data_dir, f'train_subset_{subset_ratio}.json'), 'w'))


def create_gqa_dataloaders(option):
    train_set = GQADataset(option.data_dir, filename='train_balanced_questions.json', split='train',
                           prepare_group_map=True, group_by=option.key_to_group_by)
    dataset_info = train_set
    ans_size = train_set.ans_size

    def prepare_eval_dset(filename, split):
        return GQADataset(option.data_dir, filename=filename, split=split,
                          group_name_to_ix=train_set.group_name_to_ix, group_ix_to_name=train_set.group_ix_to_name)

    # Original GQA
    val_non_ood = prepare_eval_dset('val_balanced_questions.json', 'val_non_ood')
    testdev_non_ood = prepare_eval_dset('testdev_balanced_questions.json', 'testdev_non_ood')
    # OOD splits
    val_all = prepare_eval_dset('ood_val_all.json', 'val_all')
    val_head = prepare_eval_dset('ood_val_head.json', 'val_head')
    val_tail = prepare_eval_dset('ood_val_tail.json', 'val_tail')
    testdev_all = prepare_eval_dset('ood_testdev_all.json', 'testdev_all')
    testdev_head = prepare_eval_dset('ood_testdev_head.json', 'testdev_head')
    testdev_tail = prepare_eval_dset('ood_testdev_tail.json', 'testdev_tail')

    val_alpha_sets = []
    # if option.use_varying_degrees:
    #     alpha_list = [9.0, 7.0, 5.0, 3.6, 2.8, 2.2, 1.8, 1.4, 1.0, 0.8, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4,
    #                   -0.5, -0.6, -0.7]
    #     for alpha in alpha_list:
    #         split = f'val_bal_tail_{alpha}'
    #
    #         val_alpha_sets.append(prepare_eval_dset(f'{split}.json'), split)

    def prepare_eval_loader(dset):
        return DataLoader(dset,
                          batch_size=option.batch_size,
                          shuffle=False,
                          num_workers=option.num_workers,
                          collate_fn=dict_collate_fn())

    if hasattr(option, 'train_ratio') and option.train_ratio is not None and option.train_ratio < 1:
        subset_file = os.path.join(option.data_dir, f'train_subset_{option.train_ratio}.json')
        logging.getLogger().info(f"Loading {subset_file}")
        subset_ixs = json.load(open(subset_file))
        train_set = Subset(train_set, indices=subset_ixs)

    train_loader = DataLoader(train_set,
                              batch_size=option.batch_size,
                              shuffle=True,
                              num_workers=option.num_workers,
                              collate_fn=dict_collate_fn())

    # Original GQA
    val_non_ood_loader = prepare_eval_loader(val_non_ood)
    testdev_non_ood_loader = prepare_eval_loader(testdev_non_ood)
    # OOD splits
    val_all_loader = prepare_eval_loader(val_all)
    val_head_loader = prepare_eval_loader(val_head)
    val_tail_loader = prepare_eval_loader(val_tail)
    testdev_all_loader = prepare_eval_loader(testdev_all)
    testdev_head_loader = prepare_eval_loader(testdev_head)
    testdev_tail_loader = prepare_eval_loader(testdev_tail)
    option.dataset_info = dataset_info
    option.num_classes = ans_size
    dset = train_set
    if isinstance(dset, Subset):
        dset = dset.dataset
    option.num_groups = dset.num_groups
    ret = {
        'Train': train_loader,
        'Test': {
            'Train': train_loader,
            'Val non-OOD': val_non_ood_loader,
            'Test Dev non-OOD': testdev_non_ood_loader,
            'Val All': val_all_loader,
            'Val Head': val_head_loader,
            'Val Tail': val_tail_loader,
            'Test Dev All': testdev_all_loader,
            'Test Dev Head': testdev_head_loader,
            'Test Dev Tail': testdev_tail_loader
        }
    }
    for val_set in val_alpha_sets:
        ret['Test'].append(prepare_eval_loader(val_set))
    return ret


if __name__ == "__main__":
    train_set = GQADataset(f'{ROOT}/GQA', filename='train_balanced_questions.json', split='train',
                           prepare_group_map=True, group_by='qtype_detailed', load_visual_feats=False)
    print(train_set.group_name_to_ix)
    # loader = DataLoader(train_set,
    #                     batch_size=16,
    #                     shuffle=False,
    #                     num_workers=0,
    #                     collate_fn=dict_collate_fn())
    # for batch in loader:
    #     for g in batch['local_group_name']:
    #         print(g)


# self.ques_dict[qid]
# {'semantic': [{'operation': 'select',
#    'dependencies': [],
#    'argument': 'bag (4495094)'},
#   {'operation': 'choose color',
#    'dependencies': [0],
#    'argument': 'black|yellow'}],
#  'entailed': ['17820264', '17820262'],
#  'equivalent': ['17820264', '17820262', '17820263'],
#  'question': 'Which color is the bag, black or yellow?',
#  'imageId': '2315755',
#  'isBalanced': True,
#  'groups': {'global': 'color', 'local': '10c-bag_color'},
#  'answer': 'black',
#  'semanticStr': 'select: bag (4495094)->choose color: black|yellow [0]',
#  'annotations': {'answer': {},
#   'question': {'4': '4495094'},
#   'fullAnswer': {'1': '4495094'}},
#  'types': {'detailed': 'chooseAttr',
#   'semantic': 'attr',
#   'structural': 'choose'},
#  'fullAnswer': 'The bag is black.'}
