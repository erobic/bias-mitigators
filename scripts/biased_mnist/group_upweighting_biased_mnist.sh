#!/bin/bash
source common.sh
set -e
source activate bias_mitigator

TRAINER_NAME='GroupUpweightingTrainer'
CUDA_VISIBLE_DEVICES=0 python main.py \
--expt_type biased_mnist_experiments_lr_wd \
--trainer_name ${TRAINER_NAME} \
--root_dir ${ROOT}