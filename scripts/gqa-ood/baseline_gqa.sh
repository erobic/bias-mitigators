#!/bin/bash
source common.sh
set -e
source activate bias_mitigator

TRAINER_NAME='BaseTrainer'
python main.py \
--expt_type gqa_experiments \
--lr 1e-4 \
--weight_decay 0 \
--trainer_name ${TRAINER_NAME} \
--root_dir ${ROOT}
