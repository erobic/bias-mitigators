#!/bin/bash
source common.sh
set -e
source activate bias_mitigator

TRAINER_NAME='RUBiTrainer'

python main.py \
--expt_type celebA_experiments \
--trainer_name ${TRAINER_NAME} \
--lr 1e-4 \
--weight_decay 1e-5 \
--root_dir ${ROOT}