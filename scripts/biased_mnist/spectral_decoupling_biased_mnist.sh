#!/bin/bash
source common.sh
set -e
source activate bias_mitigator

TRAINER_NAME='SpectralDecouplingTrainer'

for sd_gamma in 0.001 0.01 0.1 1 10; do
  for sd_lambda in 0.001 0.01 0.1 1 10; do
    python main.py \
    --lr 1e-3 \
    --weight_decay 0 \
    --expt_type biased_mnist_experiments \
    --trainer_name ${TRAINER_NAME} \
    --root_dir ${ROOT} \
    --spectral_decoupling_lambda ${sd_lambda} \
    --spectral_decoupling_gamma ${sd_gamma} \
    --expt_name lambda_${sd_lambda}_gamma_${sd_gamma}
  done
done