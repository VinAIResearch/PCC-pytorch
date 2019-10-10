## Embed to Control

This is a pytorch implementation of the paper "[Prediction, Consistency, Curvature: Representation Learning for Locally-Linear Control](https://arxiv.org/abs/1909.01506)".

**Note: This is not the official implementation.**

### Installing

First, clone the repository:

```
git clone https://github.com/tungnd1705/PCC-pytorch.git
```

Install the dependencies as listed in `pcc.yml` and activate the environment

```
conda env create -f pcc.yml

conda activate pcc
```

Then install the patch version of gym in order to sample the pendulum data

```
cd gym

python setup.py install
```

### Simulate training data

Currently the code supports simulating 3 environments: `planar`, `pendulum` and `cartpole`.

In order to generate data, simply run `python sample_{env_name}_data.py --sample_size={sample_size}`.

**Note: the sample size is equal to the total number of training and test data**

<!-- For the planar task, we base on [this](https://github.com/ethanluoyc/e2c-pytorch) implementation and modify for our needs. -->

### Training

Run the ``train_pcc.py`` with your own settings. E.g.,

```
python train_pcc.py \
    --env=planar \
    --armotized=False \
    --batch_size=128 \
    --lr=0.0005 \
    --decay=0.001 \
    --num_iter=5000 \
    --iter_save=1000
```

You can visualize the training process by running ``tensorboard --logdir=logs``.

### Citation

If you find PCC useful in your research, please consider citing:

```
@techreport{48535,
title	= {Prediction, Consistency, Curvature: Representation Learning for Locally-Linear Control},
author	= {Nir Levine and Yinlam Chow and Rui Shu and Ang Li and Mohammad Ghavamzadeh and Hung Bui},
year	= {2019}
}
```