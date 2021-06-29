#!/bin/bash
source common.sh
set -e
source activate bias_mitigator

TRAINER_NAME='RUBiTrainer'

python main.py \
--expt_type biased_mnist_experiments \
--trainer_name ${TRAINER_NAME} \
--lr 1e-3 \
--weight_decay 1e-3 \
--root_dir ${ROOT}