#!/bin/bash
source common.sh
set -e
source activate bias_mitigator

TRAINER_NAME='LNLTrainer'
for expt_type in biased_mnist_individual_variables biased_mnist_experiments_hierarchical biased_mnist_experiments_p_bias; do
    python main.py \
    --expt_type ${expt_type} \
    --lr 1e-4 \
    --weight_decay 0 \
    --trainer_name ${TRAINER_NAME} \
    --root_dir ${ROOT} \
    --entropy_loss_weight 0.01 \
    --grad_reverse_factor -0.001
done