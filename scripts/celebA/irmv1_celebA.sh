#!/bin/bash
source common.sh
set -e
source activate bias_mitigator

TRAINER_NAME='IRMv1Trainer'
python main.py \
--lr 1e-4 \
--weight_decay 0 \
--expt_type celebA_experiments \
--trainer_name ${TRAINER_NAME} \
--grad_penalty_weight 1 \
--num_envs_per_batch 4 \
--root_dir ${ROOT}
