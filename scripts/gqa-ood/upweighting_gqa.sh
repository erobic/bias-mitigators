#!/bin/bash
source common.sh
set -e
source activate bias_mitigator

#for key_to_group_by in head_tail answer global_group_name local_group_name; do
for key_to_group_by in head_tail ; do
  TRAINER_NAME='GroupUpweightingTrainer'
  python main.py \
  --expt_type gqa_experiments \
  --trainer_name ${TRAINER_NAME} \
  --key_to_group_by ${key_to_group_by} \
  --lr 1e-3 \
  --weight_decay 0 \
  --root_dir ${ROOT}
done