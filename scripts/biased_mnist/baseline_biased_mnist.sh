#!/bin/bash
source common.sh
set -e
source activate bias_mitigator

TRAINER_NAME='BaseTrainer'

#python main.py \
#--expt_type biased_mnist_experiments_lr_wd \
#--trainer_name ${TRAINER_NAME} \
#--root_dir ${ROOT}

python main.py \
--expt_type biased_mnist_experiments_p_bias \
--trainer_name ${TRAINER_NAME} \
--lr 1e-3 \
--weight_decay 1e-3 \
--root_dir ${ROOT}