#!/bin/bash
source common.sh
set -e
source activate bias_mitigator

TRAINER_NAME='IRMv1Trainer'

for expt_type in biased_mnist_individual_variables biased_mnist_experiments_hierarchical biased_mnist_experiments_p_bias; do
    python main.py \
    --lr 1e-3 \
    --weight_decay 0.1 \
    --expt_type ${expt_type} \
    --trainer_name ${TRAINER_NAME} \
    --grad_penalty_weight 0.01 \
    --num_envs_per_batch 16 \
    --root_dir ${ROOT}
done