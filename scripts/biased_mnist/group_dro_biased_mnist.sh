#!/bin/bash
source common.sh
set -e
source activate bias_mitigator

TRAINER_NAME='GroupDROTrainer'

for step_size in 0.1 0.01 0.001; do
  python main.py \
  --expt_type biased_mnist_experiments \
  --trainer_name ${TRAINER_NAME} \
  --lr 1e-3 \
  --weight_decay 0 \
  --group_weight_step_size ${step_size} \
  --root_dir ${ROOT} \
  --expt_name step_size_${step_size}
done