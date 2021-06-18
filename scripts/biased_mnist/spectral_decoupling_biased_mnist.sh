#!/bin/bash
source common.sh
set -e
source activate bias_mitigator

TRAINER_NAME='SpectralDecouplingTrainer'

# TODO: specify gamma/lambda
python main.py \
--lr 1e-4 \
--weight_decay 1e-5 \
--expt_type biased_mnist_experiments \
--trainer_name ${TRAINER_NAME} \
--root_dir ${ROOT} \
--spectral_decoupling_lambda 0.001 \
--spectral_decoupling_gamma 0.001
