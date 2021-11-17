#!/bin/bash
source common.sh
set -e
source activate bias_mitigator

TRAINER_NAME='SpectralDecouplingTrainer'

#for expt_type in biased_mnist_experiments_lr_wd; do
#    python main.py \
#    --expt_type ${expt_type} \
#    --trainer_name ${TRAINER_NAME} \
#    --root_dir ${ROOT} \
#    --spectral_decoupling_lambda 0.01 \
#    --spectral_decoupling_gamma 0.01
#done
for expt_type in biased_mnist_experiments_p_bias; do
    python main.py \
    --lr 1e-3 \
    --weight_decay 1e-3 \
    --expt_type ${expt_type} \
    --trainer_name ${TRAINER_NAME} \
    --root_dir ${ROOT} \
    --spectral_decoupling_lambda 0.01 \
    --spectral_decoupling_gamma 0.01
done