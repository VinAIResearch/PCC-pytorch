"""Markov Decision Process for Visual Cartpole."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from PIL import Image
from PIL import ImageDraw
import os
from pathlib import Path

root_path = str(Path(os.path.dirname(os.path.abspath(__file__))).parent)
os.sys.path.append(root_path)

from mdp.cartpole_base import CartPoleBase
from mdp.common import StateIndex


class VisualCartPoleBalance(CartPoleBase):
    """VisualCartPoleBalance."""
    # MDP goal boundaries.
    goal_limits = [-np.pi/10, np.pi/10]

    # MDP boundaries.
    angle_limits = [-np.pi, np.pi]
    angular_rate_limits = [-2 * np.pi, 2 * np.pi]
    position_limits = [-2.4, 2.4]
    velocity_limits = [-6.0, 6.0]

    # Sampling boundaries.
    angle_limits_samp = 2 * goal_limits

    def __init__(self, width=80, height=80, frequency=50,
                 noise=0.0, line_width=6):
        """Init the MDP.

        Args:
          width: integer indicating the width of the rendered image.
          height: integer indicating the height of the rendered image.
          frequency: float indicating the simulator frequency (discrete steps).
          noise: float magnitude of additive noise to the transitions.
          line_width: width of the pendulum in the rendered image.
        """
        super(VisualCartPoleBalance, self).__init__()
        self.width = width
        self.height = height
        self.dt = 1 / frequency
        self.noise = noise
        self.pole_length = height * 0.65
        self.cart_size = (width / 10., height / 20.)
        self.im = Image.new('L', (width, height))
        self.draw = ImageDraw.Draw(self.im)
        self.action_dim = 1
        self.line_width = line_width

    def render(self, s):
        """Renders the image."""
        # Draw background.
        self.draw.rectangle((0, 0, self.width, self.height), fill=0)

        # Cart location.
        base_x = self.im.size[0] / 2.0
        base_y = self.im.size[1] - 2 * self.cart_size[1] - 2
        cart_x = base_x + (s[StateIndex.X] / self.position_limits[1]) * (
                self.width / 2. - 1. * self.cart_size[0])

        # Pole location.
        end_x = cart_x + np.sin([s[StateIndex.THETA]]) * self.pole_length
        end_y = base_y - np.cos([s[StateIndex.THETA]]) * self.pole_length

        # Draw cart.
        self.draw.rectangle(
            (cart_x - self.cart_size[0], base_y - self.cart_size[1],
             cart_x + self.cart_size[0], base_y + self.cart_size[1]),
            fill=255)

        # Draw pole.
        self.draw.line(
            (cart_x, base_y, end_x, end_y), width=self.line_width, fill=255)

        return np.expand_dims(np.asarray(self.im) / 255.0, axis=-1)

    def transition_function(self, s, a, project_actions=True):
        """Transition function."""
        if project_actions:
            a = self.project_actions(a)
        true_state = s
        next_true = self._step_four_state(true_state, a)
        next_true += self.noise * np.random.normal(loc=0., scale=1.,
                                                   size=next_true.shape)
        # next_image = self.render(next_true)
        return next_true

    def is_fail(self, s):
        """Indicates whether the state results in failure."""
        angle = s[StateIndex.THETA]
        position = s[StateIndex.X]
        if ((self.goal_limits[0] < angle < self.goal_limits[1])
                and (self.position_limits[0] < position < self.position_limits[1])):
            return False
        else:
            return True

    def is_goal(self, s):
        """Inidicates whether the state achieves the goal."""
        angle = s[StateIndex.THETA]
        if self.goal_limits[0] < angle < self.goal_limits[1]:
            return True
        else:
            return False

    def reward_function(self, s):
        """Reward function."""
        if self.is_goal(s):
            reward = self.goal_reward
        else:
            reward = 0
        return reward

    def sample_random_state(self):
        """Sample a random state."""
        angle = np.random.uniform(self.angle_limits_samp[0],
                                  self.angle_limits_samp[1])
        angle_rate = np.random.uniform(self.angular_rate_limits[0],
                                       self.angular_rate_limits[1])
        pos = np.random.uniform(self.position_limits[0], self.position_limits[1])
        vel = np.random.uniform(self.velocity_limits[0],
                                self.velocity_limits[1])
        true_state = np.array([angle, angle_rate, pos, vel])
        return true_state
