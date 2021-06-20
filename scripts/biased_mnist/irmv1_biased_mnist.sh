#!/bin/bash
source common.sh
set -e
source activate bias_mitigator

TRAINER_NAME='IRMv1Trainer'
for grad_penalty_weight in 0.001 0.01 0.1 1 10 100; do
  python main.py \
  --lr 1e-3 \
  --weight_decay 1e-3 \
  --expt_type biased_mnist_experiments \
  --trainer_name ${TRAINER_NAME} \
  --grad_penalty_weight ${grad_penalty_weight} \
  --num_envs_per_batch 16 \
  --root_dir ${ROOT} \
  --expt_name gpw_${grad_penalty_weight}
done