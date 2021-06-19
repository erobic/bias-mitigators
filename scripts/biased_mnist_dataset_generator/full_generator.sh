#!/bin/bash
source activate bias_mitigator
python datasets/biased_mnist_generator.py \
--config_file conf/biased_mnist_generator/full.yaml \
--p_bias 0.9 \
--suffix '_0.9' \
--generate_test_set 1

for p_bias in 0.5 0.6 0.7 0.8 1.0; do
  python datasets/biased_mnist_generator.py \
  --config_file conf/biased_mnist_generator/full.yaml \
  --p_bias ${p_bias} \
  --suffix _${p_bias} \
  --generate_test_set 0
done