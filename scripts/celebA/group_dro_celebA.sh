#!/bin/bash
source common.sh
set -e
source activate bias_mitigator

TRAINER_NAME='GroupDROTrainer'

python main.py \
--expt_type celebA_experiments \
--trainer_name ${TRAINER_NAME} \
--lr 1e-5 \
--weight_decay 0.1 \
--group_weight_step_size 0.01 \
--root_dir ${ROOT}