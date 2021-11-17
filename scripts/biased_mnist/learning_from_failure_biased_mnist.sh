#!/bin/bash
source common.sh
set -e
source activate bias_mitigator

TRAINER_NAME='LffTrainer'

for bias_loss_gamma in 0.5; do
  python main.py \
  --expt_type biased_mnist_experiments \
  --lr 1e-3 \
  --weight_decay 1e-5 \
  --trainer_name ${TRAINER_NAME} \
  --optimizer_name Adam \
  --bias_loss_gamma ${bias_loss_gamma} \
  --root_dir ${ROOT}
done