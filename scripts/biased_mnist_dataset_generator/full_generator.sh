#!/bin/bash
source activate bias_mitigator
#python datasets/biased_mnist_generator.py \
#--config_file conf/biased_mnist_generator/full.yaml \
#--p_bias 0.9 \
#--suffix '_0.9' \
#--generate_test_set 1

for p_bias in 0.9 0.93 0.95 0.97 0.99 1.0; do
  python -u datasets/biased_mnist_generator.py \
  --config_file conf/biased_mnist_generator/full_v1.yaml \
  --p_bias ${p_bias} \
  --suffix _${p_bias} \
  --generate_test_set 0
done