#!/bin/bash
source common.sh
set -e
source activate bias_mitigator

TRAINER_NAME='RUBiTrainer'

#for expt_type in biased_mnist_experiments_lr_wd; do
#  python main.py \
#  --expt_type ${expt_type} \
#  --trainer_name ${TRAINER_NAME} \
#  --root_dir ${ROOT}
#done

for expt_type in biased_mnist_individual_variables biased_mnist_experiments_hierarchical biased_mnist_experiments_p_bias; do
  python main.py \
  --expt_type ${expt_type} \
  --trainer_name ${TRAINER_NAME} \
  --lr 1e-4 \
  --weight_decay 1e-3 \
  --root_dir ${ROOT}
done