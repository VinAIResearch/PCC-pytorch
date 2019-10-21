import os
from os import path
from tqdm import trange
import matplotlib.pyplot as plt
import numpy as np
import gym
import json
from datetime import datetime
import argparse

env = gym.make('Pendulum-v0').env
width, height = 48 * 2, 48

def render(state):
    # need two observations to restore the Markov property
    before1 = state
    before2 = env.step_from_state(state, np.array([0]))
    return map(env.render_state, (before1[0], before2[0]))

def sample_pendulum(sample_size, output_dir='data/pendulum', step_size=1, apply_control=True, num_shards=10):
    assert sample_size % num_shards == 0

    samples = []

    if not path.exists(output_dir):
        os.makedirs(output_dir)

    for i in trange(sample_size):
        """
        for each sample:
        - draw a random state (theta, theta dot)
        - render x_t (including 2 images)
        - draw a random action u_t and apply
        - render x_t+1 after applying u_t
        """
        # th (theta) and thdot (theta dot) represent a state in Pendulum env
        th = np.random.uniform(0, np.pi * 2)
        thdot = np.random.uniform(-8, 8)

        state = np.array([th, thdot])

        initial_state = np.copy(state)
        before1, before2 = render(state)

        # apply the same control over a few timesteps
        if apply_control:
            u = np.random.uniform(-2, 2, size=(1,))
        else:
            u = np.zeros((1,))

        for _ in range(step_size):
            state = env.step_from_state(state, u)

        after_state = np.copy(state)
        after1, after2 = render(state)

        before = np.hstack((before1, before2))
        after = np.hstack((after1, after2))

        shard_no = i // (sample_size // num_shards)

        shard_path = path.join('{:03d}-of-{:03d}'.format(shard_no, num_shards))

        if not path.exists(path.join(output_dir, shard_path)):
            os.makedirs(path.join(output_dir, shard_path))

        before_file = path.join(shard_path, 'before-{:05d}.jpg'.format(i))
        plt.imsave(path.join(output_dir, before_file), before)

        after_file = path.join(shard_path, 'after-{:05d}.jpg'.format(i))
        plt.imsave(path.join(output_dir, after_file), after)

        samples.append({
            'before_state': initial_state.tolist(),
            'after_state': after_state.tolist(),
            'before': before_file,
            'after': after_file,
            'control': u.tolist(),
        })

    with open(path.join(output_dir, 'data.json'), 'wt') as outfile:
        json.dump(
            {
                'metadata': {
                    'num_samples': sample_size,
                    'step_size': step_size,
                    'apply_control': apply_control,
                    'time_created': str(datetime.now()),
                    'version': 1
                },
                'samples': samples
            }, outfile, indent=2)

    env.viewer.close()

def main(args):
    sample_size = args.sample_size

    sample_pendulum(sample_size=sample_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='sample data')

    parser.add_argument('--sample_size', required=True, type=int, help='the number of samples')

    args = parser.parse_args()

    main(args)