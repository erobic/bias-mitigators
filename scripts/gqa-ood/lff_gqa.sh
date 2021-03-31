#!/bin/bash
source common.sh
set -e
source activate bias_mitigator

TRAINER_NAME='LffTrainer'
python main.py \
--expt_type gqa_lff \
--lr 1e-4 \
--weight_decay 0 \
--trainer_name ${TRAINER_NAME} \
--bias_loss_gamma 0.7 \
--root_dir ${ROOT}