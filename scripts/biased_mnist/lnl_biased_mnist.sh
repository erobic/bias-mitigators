#!/bin/bash
source common.sh
set -e
source activate bias_mitigator

TRAINER_NAME='LNLTrainer'
python main.py \
--expt_type biased_mnist_experiments \
--lr 1e-4 \
--weight_decay 1e-4 \
--trainer_name ${TRAINER_NAME} \
--root_dir ${ROOT} \
--entropy_loss_weight 0.01 \
--grad_reverse_factor -0.1