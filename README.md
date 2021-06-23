##  An Investigation of Critical Issues in Bias Mitigation Techniques

Our paper (https://arxiv.org/abs/2104.00170) examines if the state-of-the-art bias mitigation methods are able to perform well on more realistic settings: with multiple sources of biases, hidden biases and without access to test distributions. This repository has implementations/re-implementations for seven popular techniques.

  
### Setup

#### Install Dependencies

`conda create -n bias_mitigator python=3.7`

`source activate bias_mitigator`

`conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch`

`conda install tqdm opencv pandas`

`pip install PyYAML`

`pip install emnist`

#### Configure Path

- Edit the `ROOT` variable in `common.sh`. This directory will contain the datasets and the experimental results.

### Datasets

- For each dataset, we test on train, val and test splits. Each dataset file contains a function to create a dataloader for all of these splits.

#### Biased MNIST

The first version (`biased_mnist_full_v1`) of the Biased MNIST dataset entails digit recognition in the presence of the following bias variables:  
- Digit Color (e.g., 0s are mostly red)
- Digit Scale/Size (e.g., 9s are really large)
- Digit Position (in a 5x5 grid) (e.g., 0s are towards the top left corner)
- Texture (e.g., 0s have cross-hatched backgrounds)
- Texture Color
- Co-occurring letters (e.g., 0s co-occur with a's and 1s with b's)
- Color of co-occurring letters

We will release the images for this version by July 2021.

##### Generating Biased MNISTv1

- To generate Biased MNIST v1, run `./scripts/biased_mnist_dataset_generator/full_generator.sh` 
- It uses `conf/biased_mnist_generator/full_v1.yaml` configuration. 

##### Generating custom Biased MNIST splits

Generate your own versions by enabling/disabling bias variables and controlling the degree of bias from each variable in a `yaml` configuration file.
- You can set the flag `enabled` to `True` or `False` for each variable and specify the level of bias by setting the `p_bias` value. 

###### Example: Generating Colored MNIST
Create `colored_mnist.yaml` inside `conf/biased_mnist_generator` which contains the following configuration:

```yaml
biased_mnist_dir: /hdd/robik/biased_mnist
mnist_dir: /hdd/robik/MNIST
bias_split_name: colored_mnist
textures_dir: null

bias_config:
  digit_color:
    enabled: True
    p_bias: 0.99
    type: 'discrete'
  digit_scale:
    enabled: False
    p_bias: null
  digit_position:
    enabled: False
    p_bias: null
  texture:
    enabled: False
    p_bias: null
  texture_color:
    enabled: False
    p_bias: null
  letter:
    enabled: False
    p_bias: null
  letter_color:
    enabled: False
    p_bias: null
  natural_texture:
    enabled: False
    p_bias: null
  num_cells: 5
  class_imbalance_ratio: null
```

Then, run the generation script:
```
#!/bin/bash
source activate bias_mitigator

for p_bias in 0.995; do
  python -u datasets/biased_mnist_generator.py \
  --config_file conf/biased_mnist_generator/colored_mnist.yaml \
  --p_bias 0.995 \
  --suffix _0.995 \
  --generate_test_set 1
done
```
- Note the flag `generate_test_set` is set to 1 since we want a separate ColoredMNIST test set, which balances out colors, but does not introduce other factors e.g., textures.  

This should create the colored mnist split.


#### CelebA
- Download the dataset [from here](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8) and extract the data to `${ROOT}`
- We adapted the data loader from `https://github.com/kohpangwei/group_DRO`

#### GQA-OOD
- Download GQA (object features, spatial features and questions) [from here](https://cs.stanford.edu/people/dorarad/gqa/download.html)
- Build the GQA-OOD by following [these instructions](https://github.com/gqa-ood/GQA-OOD/tree/master/code)

- Download embeddings
`python -m spacy download en_vectors_web_lg`

-  Preprocess visual/spatial features
`./scripts/gqa-ood/preprocess_gqa.sh`

### Run the methods

We have provided a separate bash file for running each method on each dataset in the `scripts` directory. Here is a sample script: 

```bash
source activate bias_mitigator

TRAINER_NAME='BaseTrainer'
lr=1e-3
wd=0
python main.py \
--expt_type celebA_experiments \
--trainer_name ${TRAINER_NAME} \
--lr ${lr} \
--weight_decay ${wd} \
--expt_name ${TRAINER_NAME} \
--root_dir ${ROOT}
```

### Contribute!

- If you want to add more bias mitigation algorithms, simply follow one of the implementations inside `trainers` directory.


### Citation
Coming soon... 
