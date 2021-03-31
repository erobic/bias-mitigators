#!/bin/bash
source common.sh
set -e
source activate bias_mitigator

TRAINER_NAME='LffTrainer'

# Note that we used SGD for all other methods, however LFF didn't gel well with SGD despite searching for hyperparameters.
# So, we are using Adam.
# Also, we were unable to replicate the original paper's results with bias_loss_gamma = 0.7. We tuned it, and found that 0.1 worked best for our runs.

python main.py \
--expt_type celebA_experiments \
--lr 1e-4 \
--weight_decay 0 \
--trainer_name ${TRAINER_NAME} \
--optimizer_name Adam \
--bias_loss_gamma 0.1 \
--root_dir ${ROOT}

