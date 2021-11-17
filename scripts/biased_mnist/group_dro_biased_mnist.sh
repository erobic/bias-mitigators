#!/bin/bash
source common.sh
set -e
source activate bias_mitigator

TRAINER_NAME='GroupDROTrainer'

python -u main.py \
--expt_type biased_mnist_experiments \
--lr 1e-3 \
--weight_decay 1e-5 \
--trainer_name ${TRAINER_NAME} \
--group_weight_step_size 1e-3 \
--root_dir ${ROOT}