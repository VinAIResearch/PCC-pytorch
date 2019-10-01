import numpy as np
import os
from os import path
from tqdm import trange
import json
from datetime import datetime
import argparse
from PIL import Image, ImageDraw

np.random.seed(1)

width, height = 40, 40
obstacles_center = np.array([[20.5, 5.5], [20.5, 12.5], [20.5, 27.5], [20.5, 35.5], [10.5, 20.5], [30.5, 20.5]])

r_overlap = 0.5 # agent cannot be in any rectangular area with obstacles as centers and half-width = 0.5
r = 2.5 # radius of the obstacles when rendered
rw = 3 # robot half-width
rw_rendered = 2 # robot half-width when rendered
max_step_len = 3
env_path = os.path.dirname(os.path.abspath(__file__))
# env = np.load(env_path + '/env.npy')

def generate_env():
    """
    return the environment with 6 obstacles
    """
    # print ('Making the environment...')
    img_arr = np.zeros(shape=(width,height))

    img_env = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(img_env)
    for y, x in obstacles_center:
        draw.ellipse((int(x)-int(r), int(y)-int(r), int(x)+int(r), int(y)+int(r)), fill=255)
    img_env = img_env.convert('L')
    # img_env.save('env.png')

    img_arr = np.array(img_env) / 255.
    # np.save('./env.npy', img_arr)
    return img_arr

def get_pixel_location(s):
    # return the location of agent when rendered
    center_x, center_y = int(round(s[0])), int(round(s[1]))
    top = center_x - rw_rendered
    bottom = center_x + rw_rendered
    left = center_y - rw_rendered
    right = center_y + rw_rendered
    return top, bottom, left, right

def render(s):
    top, bottom, left, right = get_pixel_location(s)
    x = generate_env()
    x[top:bottom, left:right] = 1.  # robot is white on black background
    return x

def is_valid(s, u, s_next, epsilon = 0.1):
    # if the difference between the action and the actual distance between x and x_next are in range(0,epsilon)
    top, bottom, left, right = get_pixel_location(s)
    top_next, bottom_next, left_next, right_next = get_pixel_location(s_next)
    x_diff = np.array([top_next - top, left_next - left], dtype=np.float)
    return (not np.sqrt(np.sum((x_diff - u)**2)) > epsilon)

def is_colliding(s):
    """
    :param s: the continuous coordinate (x, y) of the agent center
    :return: if agent body overlaps with obstacles
    """
    if np.any([s - rw < 0, s + rw > height]):
        return True
    x, y = s[0], s[1]
    for obs in obstacles_center:
        if np.abs(obs[0] - x) <= r_overlap and np.abs(obs[1] - y) <= r_overlap:
            return True
    return False

def random_step(s):
    # draw a random step until it doesn't collidie with the obstacles
    while True:
        u = np.random.uniform(low = -max_step_len, high = max_step_len, size = 2)
        s_next = s + u
        if (not is_colliding(s_next) and is_valid(s, u, s_next)):
            return u, s_next

def sample(sample_size):
    """
    return [(s, u, s_next)]
    """
    state_samples = []
    for i in trange(sample_size, desc = 'Sampling data'):
        while True:
            s_x = np.random.uniform(low = rw, high = height - rw)
            s_y = np.random.uniform(low = rw, high = width - rw)
            s = np.array([s_x, s_y])
            if not is_colliding(s):
                break
        u, s_next = random_step(s)
        state_samples.append((s, u, s_next))
    obs_samples = [(render(s), u, render(s_next)) for s, u, s_next in state_samples]
    return state_samples, obs_samples

def write_to_file(sample_size, output_dir = './data/planar'):
    """
    write [(x, u, x_next)] to output dir
    """
    if not path.exists(output_dir):
        os.makedirs(output_dir)

    state_samples, obs_samples = sample(sample_size)

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
                    'max_distance': max_step_len,
                    'time_created': str(datetime.now()),
                    'version': 1
                },
                'samples': samples
            }, outfile, indent=2)

def main(args):
    sample_size = args.sample_size

    write_to_file(sample_size=sample_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='sample planar data')

    parser.add_argument('--sample_size', required=True, type=int, help='the number of samples')

    args = parser.parse_args()

    main(args)