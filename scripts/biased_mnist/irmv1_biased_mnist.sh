#!/bin/bash
source common.sh
set -e
source activate bias_mitigator

TRAINER_NAME='IRMv1Trainer'
for grad_penalty_weight in 0.01; do
  python main.py \
  --lr 1e-3 \
  --weight_decay 1e-3 \
  --expt_type biased_mnist_experiments_hierarchical \
  --trainer_name ${TRAINER_NAME} \
  --grad_penalty_weight ${grad_penalty_weight} \
  --num_envs_per_batch 16 \
  --root_dir ${ROOT}
done