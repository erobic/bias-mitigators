#!/bin/bash
source common.sh
set -e
source activate bias_mitigator

TRAINER_NAME='BaseTrainer'
lr=1e-3
wd=1e-5

python main.py \
--expt_type biased_mnist_experiments \
--trainer_name ${TRAINER_NAME} \
--lr ${lr} \
--weight_decay ${wd} \
--root_dir ${ROOT}