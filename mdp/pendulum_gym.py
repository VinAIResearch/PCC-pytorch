import numpy as np
from PIL import Image, ImageDraw
import os
import gym
from pathlib import Path

from mdp.common import StateIndex

root_path = str(Path(os.path.dirname(os.path.abspath(__file__))).parent)
os.sys.path.append(root_path)

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)

class PendulumGymMDP(object):
    # goal range
    goal_range = [-np.pi / 6, np.pi / 6]
    action_dim = 1
    goal_reward = 1
    def __init__(self, width=48, height=48, noise=0.0, render_width=4, g=10.0):
        """
        Args:
          width: width of the rendered image.
          height: height of the rendered image.
          noise: noise level
          render_width: width of the pendulum in the rendered image.
        """
        self.max_speed = 8
        self.max_torque = 2.
        self.dt = .05
        self.g = g
        self.m = 1.
        self.l = 1.
        self.viewer = None

        self.width = width
        self.height = height
        self.noise = noise

        self.render_width = render_width
        self.render_length = (width / 2) - 2

        super(PendulumGymMDP, self).__init__()

    def take_step(self, s, u):
        u = u.squeeze()

        th, thdot = s

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)

        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        s_next = np.array([newth, newthdot])
        return s_next

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

    def is_goal(self, s):
        """Check if the pendulum is in goal region"""
        angle = s[StateIndex.THETA]
        return self.goal_range[0] <= angle <= self.goal_range[1]

    def reward_function(self, s):
        """Reward function."""
        return int(self.is_goal(s)) * self.goal_reward

    def sample_random_state(self):
        high = np.array([np.pi, 1])
        state = np.random.uniform(low=-high, high=high)
        return state

    def sample_random_action(self):
        """Sample a random action from action range."""
        return np.array(
            [np.random.uniform(-self.max_torque, self.max_torque)])

    def sample_extreme_action(self):
        """Sample a random extreme action from action range."""
        return np.array(
            [np.random.choice([-self.max_torque, self.max_torque])])