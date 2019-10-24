import numpy as np
import os
from os import path
from tqdm import trange
import json
from datetime import datetime
import argparse
from PIL import Image, ImageDraw

np.random.seed(1)
env_path = os.path.dirname(os.path.abspath(__file__))
os.sys.path.append(env_path)
from planar_env import PlanarEnv

np.random.seed(1)

def sample(planar_env, sample_size):
    """
    return [(s, u, s_next)]
    """
    state_samples = []
    for i in trange(sample_size, desc = 'Sampling data'):
        while True:
            s = planar_env.random_state()
            if not planar_env.is_colliding(s):
                break
        u, s_next = planar_env.random_step(s)
        state_samples.append((s, u, s_next))
    obs_samples = [(planar_env.render(s), u, planar_env.render(s_next)) for s, u, s_next in state_samples]
    return state_samples, obs_samples

def write_to_file(planar_env, sample_size, output_dir = env_path + '/raw'):
    """
    write [(x, u, x_next)] to output dir
    """
    if not path.exists(output_dir):
        os.makedirs(output_dir)

    state_samples, obs_samples = sample(planar_env, sample_size)

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
                    'max_distance': planar_env.max_step_len,
                    'time_created': str(datetime.now()),
                    'version': 1
                },
                'samples': samples
            }, outfile, indent=2)

def main(args):
    sample_size = args.sample_size
    planar_env = PlanarEnv()
    write_to_file(planar_env, sample_size=sample_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='sample planar data')

    parser.add_argument('--sample_size', required=True, type=int, help='the number of samples')

    args = parser.parse_args()

    main(args)