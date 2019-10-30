import numpy as np
import os
from os import path
from pathlib import Path
from tqdm import trange
import json
from datetime import datetime
import argparse
from PIL import Image, ImageDraw

np.random.seed(1)
root_path = str(Path(os.path.dirname(os.path.abspath(__file__))).parent)
os.sys.path.append(root_path)
from mdp.plane_obstacles_mdp import PlanarObstaclesMDP

np.random.seed(1)

def sample(mdp, sample_size):
    """
    return [(s, u, s_next)]
    """
    state_samples = []
    for i in trange(sample_size, desc = 'Sampling data'):
        s = mdp.sample_valid_random_state()
        u = mdp.sample_valid_random_action(s)
        s_next = mdp.transition_function(s, u)
        state_samples.append((s, u, s_next))
    obs_samples = [(mdp.render(s), u, mdp.render(s_next)) for s, u, s_next in state_samples]
    return state_samples, obs_samples

def write_to_file(mdp, sample_size, output_dir):
    """
    write [(x, u, x_next)] to output dir
    """
    if not path.exists(output_dir):
        os.makedirs(output_dir)

    state_samples, obs_samples = sample(mdp, sample_size)

    samples = []

    for i, (before, u, after) in enumerate(obs_samples):
        before_file = 'before-{:05d}.png'.format(i)
        Image.fromarray(before * 255.).convert('L').save(path.join(output_dir, before_file))

        after_file = 'after-{:05d}.png'.format(i)
        Image.fromarray(after * 255.).convert('L').save(path.join(output_dir, after_file))

        initial_state = state_samples[i][0]
        after_state = state_samples[i][2]

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
                    'max_distance': mdp.max_step,
                    'time_created': str(datetime.now()),
                    'version': 1
                },
                'samples': samples
            }, outfile, indent=2)

def main(args):
    sample_size = args.sample_size
    noise = args.noise
    mdp = PlanarObstaclesMDP(sampling=True, noise = noise)
    write_to_file(mdp, sample_size, root_path + '/data/planar/raw')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='sample planar data')

    parser.add_argument('--sample_size', required=True, type=int, help='the number of samples')
    parser.add_argument('--noise', default=0, type=int, help='level of noise')

    args = parser.parse_args()

    main(args)