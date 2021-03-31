#!/bin/bash
source common.sh
set -e
source activate bias_mitigator

TRAINER_NAME='LNLTrainer'
#BIAS_VARIABLE_NAME=answer
#BIAS_VARIABLE_NAME=head_tail

for key_to_group_by in qtype_detailed; do
  EXPT_NAME=bias_variable_${key_to_group_by}
  python main.py \
  --expt_type gqa_experiments \
  --lr 1e-3 \
  --weight_decay 0 \
  --grad_reverse_factor -0.1 \
  --entropy_loss_weight 0.01 \
  --key_to_group_by ${key_to_group_by} \
  --bias_variable_name group_ix \
  --trainer_name ${TRAINER_NAME} \
  --expt_name ${EXPT_NAME} \
  --root_dir ${ROOT}
done