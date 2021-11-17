#!/bin/bash
source common.sh
set -e
source activate bias_mitigator

TRAINER_NAME='LNLTrainer'
for entropy_loss_weight in 0.01; do
  for grad_reverse_factor in -0.1; do
    python main.py \
    --expt_type biased_mnist_experiments \
    --lr 1e-3 \
    --weight_decay 1e-5 \
    --trainer_name ${TRAINER_NAME} \
    --root_dir ${ROOT} \
    --entropy_loss_weight ${entropy_loss_weight} \
    --grad_reverse_factor ${grad_reverse_factor}
  done
done