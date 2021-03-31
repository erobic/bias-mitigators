#!/bin/bash
source common.sh
set -e
source activate bias_mitigator

TRAINER_NAME='BaseTrainer'
lr=1e-3
wd=0
python main.py \
--expt_type celebA_experiments \
--trainer_name ${TRAINER_NAME} \
--lr ${lr} \
--weight_decay ${wd} \
--expt_name ${TRAINER_NAME} \
--root_dir ${ROOT}