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

class PlanarEnv(object):
    width = 40
    height = 40
    obstacles_center = np.array([[20.5, 5.5], [20.5, 12.5], [20.5, 27.5], [20.5, 35.5], [10.5, 20.5], [30.5, 20.5]])

    r_overlap = 0.5 # agent cannot be in any rectangular area with obstacles as centers and half-width = 0.5
    r = 2.5 # radius of the obstacles when rendered
    rw = 3 # robot half-width
    rw_rendered = 2 # robot half-width when rendered
    max_step_len = 3

    def __init__(self):
        super(PlanarEnv, self).__init__()

    def random_state(self):
        s_x = np.random.uniform(low = self.rw, high = self.height - self.rw)
        s_y = np.random.uniform(low = self.rw, high = self.width - self.rw)
        s = np.array([s_x, s_y])
        return s

    def random_step(self, s):
        # draw a random step until it doesn't collidie with the obstacles
        while True:
            u = np.random.uniform(low = -self.max_step_len, high = self.max_step_len, size = 2)
            s_next = s + u
            if (not self.is_colliding(s_next) and self.is_valid(s, u, s_next)):
                return u, s_next

    def is_valid(self, s, u, s_next, epsilon = 0.1):
        # if the difference between the action and the actual distance between x and x_next are in range(0,epsilon)
        top, bottom, left, right = self.get_pixel_location(s)
        top_next, bottom_next, left_next, right_next = self.get_pixel_location(s_next)
        x_diff = np.array([top_next - top, left_next - left], dtype=np.float)
        return (not np.sqrt(np.sum((x_diff - u)**2)) > epsilon)

    def is_colliding(self, s):
        """
        :param s: the continuous coordinate (x, y) of the agent center
        :return: if agent body overlaps with obstacles
        """
        if np.any([s - self.rw < 0, s + self.rw > self.height]):
            return True
        x, y = s[0], s[1]
        for obs in self.obstacles_center:
            if np.abs(obs[0] - x) <= self.r_overlap and np.abs(obs[1] - y) <= self.r_overlap:
                return True
        return False

    def render(self, s):
        top, bottom, left, right = self.get_pixel_location(s)
        x = self.generate_env()
        x[top:bottom, left:right] = 1.  # robot is white on black background
        return x

    def generate_env(self):
        """
        return the environment with 6 obstacles
        """
        # print ('Making the environment...')
        img_arr = np.zeros(shape=(self.width, self.height))

        img_env = Image.fromarray(img_arr)
        draw = ImageDraw.Draw(img_env)
        for y, x in self.obstacles_center:
            draw.ellipse((int(x)-int(self.r), int(y)-int(self.r), int(x)+int(self.r), int(y)+int(self.r)), fill=255)
        img_env = img_env.convert('L')

        img_arr = np.array(img_env) / 255.
        return img_arr

    def get_pixel_location(self, s):
        # return the location of agent when rendered
        center_x, center_y = int(round(s[0])), int(round(s[1]))
        top = center_x - self.rw_rendered
        bottom = center_x + self.rw_rendered
        left = center_y - self.rw_rendered
        right = center_y + self.rw_rendered
        return top, bottom, left, right


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

def write_to_file(planar_env, sample_size, output_dir = env_path + '/raw_test'):
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