#!/bin/bash
source common.sh
set -e
source activate bias_mitigator

TRAINER_NAME='GroupDROTrainer'

for step_size in 0.001; do
  CUDA_VISIBLE_DEVICES=0 python main.py \
  --expt_type biased_mnist_experiments_lr_wd \
  --trainer_name ${TRAINER_NAME} \
  --group_weight_step_size ${step_size} \
  --root_dir ${ROOT}
done