#!/bin/bash
source common.sh
set -e
source activate bias_mitigator

TRAINER_NAME='LNLTrainer'
python main.py \
--expt_type celebA_experiments \
--lr 1e-4 \
--weight_decay 1e-4 \
--trainer_name ${TRAINER_NAME} \
--root_dir ${ROOT}

#TRAINER_NAME='LNLTrainer'
#python main.py \
#--expt_type celebA_experiments \
#--trainer_name ${TRAINER_NAME} \
#--root_dir ${ROOT}