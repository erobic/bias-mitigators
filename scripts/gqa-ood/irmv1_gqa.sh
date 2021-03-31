#!/bin/bash
source common.sh
set -e
source activate bias_mitigator

TRAINER_NAME='IRMv1Trainer'

#for key_to_group_by in head_tail answer global_group_name local_group_name; do
for key_to_group_by in head_tail ; do
    python main.py \
    --lr 1e-4 \
    --weight_decay 0 \
    --expt_type gqa_experiments \
    --key_to_group_by ${key_to_group_by} \
    --trainer_name ${TRAINER_NAME} \
    --grad_penalty_weight 0.01 \
    --num_envs_per_batch 16 \
    --root_dir ${ROOT}
done