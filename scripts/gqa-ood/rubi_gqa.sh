#!/bin/bash
source common.sh
set -e
source activate bias_mitigator

TRAINER_NAME='RUBiTrainer'
#for key_to_group_by in head_tail global_group_name local_group_name; do
for key_to_group_by in head_tail ; do
  python main.py \
  --expt_type gqa_experiments \
  --lr 1e-4 \
  --weight_decay 0 \
  --trainer_name ${TRAINER_NAME} \
  --key_to_group_by ${key_to_group_by} \
  --bias_variable_name group_ix \
  --root_dir ${ROOT}
done
