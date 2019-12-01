import numpy as np
from PIL import Image, ImageDraw
import os
import gym
from pathlib import Path

from mdp.common import StateIndex

root_path = str(Path(os.path.dirname(os.path.abspath(__file__))).parent)
os.sys.path.append(root_path)

class PendulumGymMDP(object):
    # goal range
    goal_range = [-np.pi / 6, np.pi / 6]
    action_dim = 1
    def __init__(self, base_mdp=gym.make("Pendulum-v0").env,
                 width=48, height=48, noise=0.0, render_width=4):
        """
        Args:
          width: width of the rendered image.
          height: height of the rendered image.
          noise: noise level
          render_width: width of the pendulum in the rendered image.
        """
        self.base_mdp = base_mdp

        self.width = width
        self.height = height
        self.noise = noise

        self.render_width = render_width
        self.render_length = (width / 2) - 2

        super(PendulumGymMDP, self).__init__()

    def take_step(self, s, u):
        return self.base_mdp.step_from_state(s, u)

    def transition_function(self, s, u): # compute next state and add noise
        s_next = self.take_step(s, u)
        # add noise
        s_next += self.noise * np.random.rand(*s_next.shape)
        return s_next

    def render(self, s):
        im = Image.new('L', (self.width, self.height))
        draw = ImageDraw.Draw(im)
        # black background
        draw.rectangle((0, 0, self.width, self.height), fill=0)

        # pendulum location.
        x_center = im.size[0] / 2.
        y_center = im.size[1] / 2.
        x_end = x_center + np.sin(s[0]) * self.render_length
        y_end = y_center - np.cos(s[0]) * self.render_length

        # white pendulum
        draw.line((x_center, y_center, x_end, y_end), width=self.render_width, fill=255)

        return np.expand_dims(np.asarray(im) / 255.0, axis=-1)

    def is_fail(self, s):
        return False

    def sample_random_state(self):
        self.base_mdp.reset()
        return self.base_mdp.state

    def sample_random_action(self):
        return np.random.uniform(-self.base_mdp.max_torque, self.base_mdp.max_torque, size=self.action_dim)

    def sample_extreme_action(self):
        return np.random.choice([-self.base_mdp.max_torque, self.base_mdp.max_torque])