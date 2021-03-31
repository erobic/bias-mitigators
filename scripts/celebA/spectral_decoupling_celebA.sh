#!/bin/bash
source common.sh
set -e
source activate bias_mitigator

TRAINER_NAME='SpectralDecouplingTrainer'

# Lambdas and gammas are specified in celebA_experiments.py
python main.py \
--lr 1e-4 \
--weight_decay 1e-5 \
--expt_type celebA_experiments \
--trainer_name ${TRAINER_NAME} \
--root_dir ${ROOT}
