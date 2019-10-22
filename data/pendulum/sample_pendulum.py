import os
from os import path
from tqdm import trange
import matplotlib.pyplot as plt
import numpy as np
import gym
import json
from datetime import datetime
import argparse

np.random.seed(1)
env_path = os.path.dirname(os.path.abspath(__file__))

width = 48 * 2
height = 48

def render(env, state):
    # need two observations to restore the Markov property
    before1 = state
    before2 = env.step_from_state(state, np.array([0]))
    return map(env.render_state, (before1[0], before2[0]))

def sample(pendulum_env, sample_size, step_size=1):
    state_samples = []
    obs_samples = []

    for i in trange(sample_size):
        # random initial state
        th = np.random.uniform(0, np.pi * 2)
        thdot = np.random.uniform(-8, 8)

        state = np.array([th, thdot])
        initial_state = np.copy(state)

        # apply the same control over a few timesteps
        u = np.random.uniform(-2, 2, size=(1,))

        for i in range(step_size):
            state = pendulum_env.step_from_state(state, u)

        after_state = np.copy(state)
        state_samples.append((initial_state, u, after_state))

        before1, before2 = render(pendulum_env, initial_state)
        after1, after2 = render(pendulum_env, after_state)

        before = np.hstack((before1, before2))
        after = np.hstack((after1, after2))
        obs_samples.append((before, u, after))

    pendulum_env.viewer.close()
    return state_samples, obs_samples

def write_to_file(pendulum_env, sample_size, step_size=1, output_dir = env_path + '/raw'):
    if not path.exists(output_dir):
        os.makedirs(output_dir)

    state_samples, obs_samples = sample(pendulum_env, sample_size, step_size)
    samples = []

    for i in range(sample_size):
        initial_state = state_samples[i][0]
        u = state_samples[i][1]
        after_state = state_samples[i][2]
        before = obs_samples[i][0]
        after = obs_samples[i][2]

        before_file = path.join(output_dir, 'before-{:05d}.jpg'.format(i))
        plt.imsave(before_file, before)

        after_file = path.join(output_dir, 'after-{:05d}.jpg'.format(i))
        plt.imsave(after_file, after)

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
                    'time_created': str(datetime.now()),
                    'version': 1
                },
                'samples': samples
            }, outfile, indent=2)

def main(args):
    sample_size = args.sample_size

    env = gym.make('Pendulum-v0').env
    write_to_file(env, sample_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='sample data')

    parser.add_argument('--sample_size', required=True, type=int, help='the number of samples')

    args = parser.parse_args()

    main(args)
