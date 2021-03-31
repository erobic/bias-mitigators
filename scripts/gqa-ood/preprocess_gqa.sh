#!/bin/bash
source common.sh
set -e
source activate bias_mitigator

mkdir -p ${ROOT}/GQA/preprocessed/objects
mkdir -p ${ROOT}/GQA/preprocessed/spatial

python datasets/vqa/gqa_feat_preproc.py \
--mode object \
--object_dir ${ROOT}/GQA/objects \
--out_dir ${ROOT}/GQA/preprocessed/objects

python datasets/vqa/gqa_feat_preproc.py \
--mode spatial \
--spatial_dir ${ROOT}/GQA/spatial \
--out_dir ${ROOT}/GQA/preprocessed/spatial
