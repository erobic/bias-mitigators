#!/bin/bash
source common.sh
set -e
source activate bias_mitigator

TRAINER_NAME='SpectralDecouplingTrainer'

for sd_gamma in 1e-3; do
  for sd_lambda in 1e-3; do
    python main.py \
    --lr 1e-3 \
    --weight_decay 1e-5 \
    --expt_type biased_mnist_experiments \
    --trainer_name ${TRAINER_NAME} \
    --root_dir ${ROOT} \
    --spectral_decoupling_lambda ${sd_lambda} \
    --spectral_decoupling_gamma ${sd_gamma}
  done
done