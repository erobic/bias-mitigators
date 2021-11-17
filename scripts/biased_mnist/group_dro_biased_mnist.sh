#!/bin/bash
source common.sh
set -e
source activate bias_mitigator

TRAINER_NAME='GroupDROTrainer'

#for expt_type in biased_mnist_experiments_lr_wd; do
#  python -u main.py \
#  --expt_type ${expt_type} \
#  --trainer_name ${TRAINER_NAME} \
#  --group_weight_step_size 0.001 \
#  --root_dir ${ROOT}
#done

for expt_type in biased_mnist_individual_variables biased_mnist_experiments_hierarchical biased_mnist_experiments_p_bias; do
  python -u main.py \
  --expt_type ${expt_type} \
  --lr 1e-4 \
  --weight_decay 1e-3 \
  --trainer_name ${TRAINER_NAME} \
  --group_weight_step_size 0.001 \
  --root_dir ${ROOT}
done