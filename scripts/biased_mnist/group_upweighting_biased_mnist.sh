#!/bin/bash
source common.sh
set -e
source activate bias_mitigator

TRAINER_NAME='GroupUpweightingTrainer'
python main.py \
--expt_type biased_mnist_experiments \
--lr 1e-5 \
--weight_decay 0.1 \
--trainer_name ${TRAINER_NAME} \
--root_dir ${ROOT}