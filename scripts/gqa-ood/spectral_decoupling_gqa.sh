#!/bin/bash
source common.sh
set -e
source activate bias_mitigator

TRAINER_NAME='SpectralDecouplingTrainer'

python main.py \
--lr 1e-4 \
--weight_decay 0 \
--expt_type gqa_sd_experiments \
--trainer_name ${TRAINER_NAME} \
--root_dir ${ROOT} \
--spectral_decoupling_lambda 1e-3 \
--spectral_decoupling_gamma 1e-3