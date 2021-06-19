#!/bin/bash
source common.sh
set -e
source activate bias_mitigator

TRAINER_NAME='LffTrainer'

python main.py \
--expt_type biased_mnist_experiments \
--lr 1e-4 \
--weight_decay 0 \
--trainer_name ${TRAINER_NAME} \
--optimizer_name Adam \
--bias_loss_gamma 0.5 \
--root_dir ${ROOT}

