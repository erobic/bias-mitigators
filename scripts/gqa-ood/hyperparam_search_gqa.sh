#!/bin/bash
source common.sh
set -e
source activate bias_mitigator
PROJECT_NAME='GQA'

TRAINER_NAME='BaseTrainer'
for lr in 1e-2; do
    python main.py \
    --expt_type gqa_experiments \
    --project_name ${PROJECT_NAME} \
    --trainer_name ${TRAINER_NAME} \
    --lr ${lr} \
    --weight_decay 0 \
    --root_dir ${ROOT}
done

#TRAINER_NAME='RUBiTrainer'
#for lr in 1e-2; do
#    for wd in 0 1e-3 0.1; do
#      python main.py \
#      --expt_type gqa_experiments \
#      --project_name ${PROJECT_NAME} \
#      --trainer_name ${TRAINER_NAME} \
#      --lr ${lr} \
#      --weight_decay ${wd} \
#      --root_dir ${ROOT}
#    done
#done


#TRAINER_NAME='GroupDROTrainer'
#for lr in 1e-2; do
#  for wd in 0 1e-3 0.1; do
#    python main.py \
#    --expt_type gqa_experiments \
#    --project_name ${PROJECT_NAME} \
#    --trainer_name ${TRAINER_NAME} \
#    --lr ${lr} \
#    --weight_decay ${wd} \
#    --root_dir ${ROOT}
#  done
#done

#TRAINER_NAME='LffTrainer'
#for lr in 1e-2; do
#    for wd in 0 1e-3 0.1; do
#      python main.py \
#      --expt_type gqa_experiments \
#      --project_name ${PROJECT_NAME} \
#      --trainer_name ${TRAINER_NAME} \
#      --lr ${lr} \
#      --weight_decay ${wd} \
#      --root_dir ${ROOT}
#    done
#done


