import numpy as np
import os
from os import path
from pathlib import Path
from tqdm import trange
import json
from datetime import datetime
import argparse
from PIL import Image, ImageDraw

root_path = str(Path(os.path.dirname(os.path.abspath(__file__))).parent)
os.sys.path.append(root_path)
from mdp.plane_obstacles_mdp import PlanarObstaclesMDP

np.random.seed(1)

def get_all_pos(mdp):
    start = 0
    end = mdp.width
    state_list = []
    for x in range(start, end):
        for y in range(start, end):
            state = np.array([x,y])
            if mdp.is_valid_state(state):
                state_list.append(state)
    return state_list

def sample(sample_size=5000, noise=0.0):
    """
    return [(s, u, s_next)]
    """
    mdp = PlanarObstaclesMDP(noise=noise)

    # place holder for data
    x_data = np.zeros((sample_size, mdp.width, mdp.height), dtype='float32')
    u_data = np.zeros((sample_size, mdp.action_dim), dtype='float32')
    x_next_data = np.zeros((sample_size, mdp.width, mdp.height), dtype='float32')
    state_data = np.zeros((sample_size, 2), dtype='float32')
    state_next_data = np.zeros((sample_size, 2), dtype='float32')

    # get all possible states (discretized on integer grid)
    state_list = get_all_pos(mdp)

    for j in trange(sample_size, desc = 'Sampling remaining data'):
        state_data[j] = state_list[j % len(state_list)]
        x_data[j] = mdp.render(state_data[j])
        u_data[j] = mdp.sample_valid_random_action(state_data[j])
        state_next_data[j] = mdp.transition_function(state_data[j], u_data[j])
        x_next_data[j] = mdp.render(state_next_data[j])
    return x_data, u_data, x_next_data, state_data, state_next_data

def write_to_file(noise, sample_size):
    """
    write [(x, u, x_next)] to output dir
    """
    output_dir = root_path + '/data/planar/raw_{:d}_{:.0f}'.format(sample_size, noise)
    if not path.exists(output_dir):
        os.makedirs(output_dir)

    x_data, u_data, x_next_data, state_data, state_next_data = sample(sample_size, noise)

    samples = []

    for i, _ in enumerate(x_data):
        before_file = 'before-{:05d}.png'.format(i)
        Image.fromarray(x_data[i] * 255.).convert('L').save(path.join(output_dir, before_file))

        after_file = 'after-{:05d}.png'.format(i)
        Image.fromarray(x_next_data[i] * 255.).convert('L').save(path.join(output_dir, after_file))

        initial_state = state_data[i]
        after_state = state_next_data[i]
        u = u_data[i]

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
                    'time_created': str(datetime.now()),
                    'version': 1
                },
                'samples': samples
            }, outfile, indent=2)

def main(args):
    sample_size = args.sample_size
    noise = args.noise
    write_to_file(noise, sample_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='sample planar data')

    parser.add_argument('--sample_size', required=True, type=int, help='the number of samples')
    parser.add_argument('--noise', default=0, type=int, help='level of noise')

    args = parser.parse_args()

    main(args)