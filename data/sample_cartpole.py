"""get_pole_simple_dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from pathlib import Path
from tqdm import trange
import os.path as path
from PIL import Image
import json
from datetime import datetime
import argparse

root_path = str(Path(os.path.dirname(os.path.abspath(__file__))).parent)
os.sys.path.append(root_path)

from mdp.cartpole_mdp import VisualCartPoleBalance

def sample(sample_size=20000, width=80, height=80, frequency=50, noise=0.0):
    """
    return [(x, u, x_next, s, s_next)]
    """
    mdp = VisualCartPoleBalance(width=width, height=height,
                                frequency=frequency, noise=noise)

    # Data buffers to fill.
    x_data = np.zeros((sample_size, width, height, 2), dtype='float32')
    u_data = np.zeros((sample_size, mdp.action_dim), dtype='float32')
    x_next_data = np.zeros((sample_size, width, height, 2), dtype='float32')
    state_data = np.zeros((sample_size, 4, 2), dtype='float32')
    state_next_data = np.zeros((sample_size, 4, 2), dtype='float32')

    # Generate interaction tuples (random states and actions).
    for sample in trange(sample_size, desc = 'Sampling data'):
        s0 = mdp.sample_random_state()
        a0 = np.atleast_1d(
            np.random.uniform(mdp.avail_force[0], mdp.avail_force[1]))
        s1 = mdp.transition_function(s0, a0)
        a1 = np.atleast_1d(
            np.random.uniform(mdp.avail_force[0], mdp.avail_force[1]))
        s2 = mdp.transition_function(s1, a1)

        ## Store interaction tuple.
        # Current state (w/ history).
        x_data[sample, :, :, 0] = s0[1][:, :, 0]
        x_data[sample, :, :, 1] = s1[1][:, :, 0]
        state_data[sample, :, 0] = s0[0][0:4]
        state_data[sample, :, 1] = s1[0][0:4]
        # Action.
        u_data[sample] = a1
        # Next state (w/ history).
        x_next_data[sample, :, :, 0] = s1[1][:, :, 0]
        x_next_data[sample, :, :, 1] = s2[1][:, :, 0]
        state_next_data[sample, :, 0] = s1[0][0:4]
        state_next_data[sample, :, 1] = s2[0][0:4]

    return x_data, u_data, x_next_data, state_data, state_next_data

def write_to_file(data, output_dir):
    """
    write [(x, u, x_next)] to output dir
    """
    if not path.exists(output_dir):
        os.makedirs(output_dir)

    samples = []
    x_data, u_data, x_next_data, state_data, state_next_data = data

    for i in range(x_data.shape[0]):
        x_1 = x_data[i, :, :, 0]
        x_2 = x_data[i, :, :, 1]
        before = np.hstack((x_1, x_2))
        before_file = 'before-{:05d}.png'.format(i)
        Image.fromarray(before * 255.).convert('L').save(path.join(output_dir, before_file))

        after_file = 'after-{:05d}.png'.format(i)
        x_next_1 = x_next_data[i, :, :, 0]
        x_next_2 = x_next_data[i, :, :, 1]
        after = np.hstack((x_next_1, x_next_2))
        Image.fromarray(after * 255.).convert('L').save(path.join(output_dir, after_file))

        initial_state = state_data[i]
        after_state = state_next_data[i]

        samples.append({
            'before_state': initial_state.tolist(),
            'after_state': after_state.tolist(),
            'before': before_file,
            'after': after_file,
            'control': u_data[i].tolist(),
        })

    with open(path.join(output_dir, 'data.json'), 'wt') as outfile:
        json.dump(
            {
                'metadata': {
                    'num_samples': x_data.shape[0],
                    'time_created': str(datetime.now()),
                    'version': 1
                },
                'samples': samples
            }, outfile, indent=2)

# data = sample(sample_size=1, width=48, height=48, frequency=50, noise=0.0)
def main(args):
    sample_size = args.sample_size
    noise = args.noise
    data = sample(sample_size=sample_size, width=80, height=80, frequency=50, noise=noise)
    write_to_file(data, root_path + '/data/cartpole/raw')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='sample cartpole data')

    parser.add_argument('--sample_size', required=True, type=int, help='the number of samples')
    parser.add_argument('--noise', default=0, type=int, help='level of noise')

    args = parser.parse_args()

    main(args)