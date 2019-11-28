## Prediction, Consistency and Curvature

This is a pytorch implementation of the paper "[Prediction, Consistency, Curvature: Representation Learning for Locally-Linear Control](https://arxiv.org/abs/1909.01506)". The work was done during the residency at [VinAI Research](vinai.io), Hanoi, Vietnam.

## Installing

First, clone the repository:

```
git clone https://github.com/tungnd1705/PCC-pytorch.git
```

Then install the dependencies as listed in `pcc.yml` and activate the environment:

```
conda env create -f pcc.yml

conda activate pcc
```

## Training

The code currently supports training for ``planar`` and ``pendulum`` environment. Run the ``train_pcc.py`` with your own settings. For example:

```
python train_pcc.py \
    --env=planar \
    --armotized=False \
    --log_dir=planar_1 \
    --seed=1 \
    --data_size = 5000 \
    --noise = 0 \
    --batch_size=128 \
    --lam_p=1.0 \
    --lam_c=8.0 \
    --lam_cur=8.0 \
    --vae_coeff=0.01 \
    --determ_coeff=0.3 \
    --lr=0.0005 \
    --decay=0.001 \
    --num_iter=5000 \
    --iter_save=1000 \
    --save_map=True
```

First, data is sampled according to the given data size and noise level, then PCC model will be trained using the specified settings.

If the argument save_map is set to True, the latent map will be drawn every 10 epoches (for planar only), then the gif file will be saved at the  same directory as the trained model.

You can also visualize the training process by running ``tensorboard --logdir={path_to_log_file}``, where ``path_to_log_file`` has the form ``logs/{env}/{log_dir}``. The trained model will be saved at ``result/{env}/{log_dir}``.

### Visualizing latent maps

You can visualize the latent map for both planar and pendulum, to do that simply run:

```
python latent_map_planar.py --log_path={log_to_trained_model} --epoch={epoch}
or 
python latent_map_pendulum.py --log_path={log_to_trained_model} --epoch={epoch}
```

## Sampling data

You can generate the training images for visualization by simply running:

```
cd data

python sample_{env_name}_data.py --sample_size={sample_size} --noise={noise}
```

Currently the code supports simulating 3 environments: `planar`, `pendulum` and `cartpole`. The MDP for `pendulum` and `cartpole` is based on Yinlam's code.

The raw data (images) is saved in data/{env_name}/raw\_{noise}\_noise

## Running iLQR on latent space

The configuration file for running iLQR for each task is in ``ilqr_config`` folder, you can modify with your own settings. Run ``python ilqr.py --task={task}``, where ``task`` is in ``{plane, swing, balance, cartpole}``.

The code will run iLQR for all models trained for that specific task and compute some statistics. The result is saved in ``iLQR/result``.

## Result
We evaluate the PCC model in 2 ways: quality of the latent map and the percentage of time the agent spent in the goal region.
### Planar system

#### Latent map
Below is a random latent map PCC produces. You can watch a video clip comparing how latent maps produced by E2C and PCC evolve at this link: https://www.youtube.com/watch?v=pBmzFvvE2bo.

![Latent space learned by PCC](sample_results/latent_map_sample.png)

#### Control result
We got around 41% on average and around 76% for the best model. Below are 2 sample trajectories of the agent.

![Sample planar trajectory 1](sample_results/planar_1.gif)

![Sample planar trajectory 2](sample_results/planar_2.gif)

### Inverted pendulum
We got around 51% on average and around 86.4% for the best model. Below are 2 sample trajectories of the inverted pendulum.

![Sample inverted pendulum trajectory 1](sample_results/pendulum_1.gif)

![Sample inverted pendulum trajectory 2](sample_results/pendulum_2.gif)
### Cartpole

![Sample cartpole trajectory 1](sample_results/cartpole_1.gif)

![Sample cartpole trajectory 2](sample_results/cartpole_2.gif)

### Acknowledgment

Many thanks to Nir Levine and Yinlam Chow for their help in answering the questions related to the PCC paper.

### Citation

If you find this implementation useful for your work, please consider starring this repository.
