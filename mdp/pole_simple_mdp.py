"""Simple Pole MDP."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy.integrate import ode as rk4
import numpy as np
from PIL import Image
from PIL import ImageDraw
from scipy.integrate import solve_ivp
import os
from pathlib import Path

root_path = str(Path(os.path.dirname(os.path.abspath(__file__))).parent)
os.sys.path.append(root_path)

from mdp.cartpole_mdp import CartPoleBase
from mdp.common import bound
from mdp.common import StateIndex
from mdp.common import wrap


class VisualPoleSimpleSwingUp(CartPoleBase):
    """Visual Simple Pole Swing Up.

    Inherits dynamics from ``CartPoleBase`` and utilizes two states - the
    angular position and velocity of the pendulum on the cart.
    """
    # MDP goal boundaries
    goal_limits = [-np.pi / 6, np.pi / 6]

    # MDP boundaries.
    angle_limits = [-np.pi, np.pi]
    angular_rate_limits = [-3 * np.pi, 3 * np.pi]
    position_limits = [-np.inf, np.inf]
    velocity_limits = [-np.inf, np.inf]

    def __init__(self, width=48, height=48, frequency=50,
                 noise=0.0, torque=1.0, line_width=4):
        """Init the MDP.

        Args:
          width: integer indicating the width of the rendered image.
          height: integer indicating the height of the rendered image.
          frequency: float indicating the simulator frequency (discrete steps).
          noise: float magnitude of additive noise to the transitions.
          torque: float maximal (absolute value) torque that can be applied.
          line_width: width of the pendulum in the rendered image.
        """
        # Limits of each dimension of the state space.
        self.statespace_limits = np.array(
            [self.angle_limits, self.angular_rate_limits])
        self.continuous_dims = [StateIndex.THETA, StateIndex.THETA_DOT]

        super(VisualPoleSimpleSwingUp, self).__init__()

        self.height = height
        self.width = width
        self.dt = 1.0 / frequency
        self.noise = noise
        self.avail_force = np.array([-torque, torque])
        self.im = Image.new('L', (width, height))
        self.draw = ImageDraw.Draw(self.im)
        self.line_width = line_width
        self.action_dim = 1
        self.visual_length = (width / 2) - 2

    def render(self, s):
        """Renders the image."""
        # Draw background.
        self.draw.rectangle((0, 0, self.width, self.height), fill=0)

        # Pole location.
        centerx = self.im.size[0] / 2.
        centery = self.im.size[1] / 2.
        xend = centerx + np.sin(s[0]) * self.visual_length
        yend = centery - np.cos(s[0]) * self.visual_length

        # Draw pole.
        self.draw.line(
            (centerx, centery, xend, yend), width=self.line_width, fill=255)

        return np.expand_dims(np.asarray(self.im) / 255.0, axis=-1)

    def transition_function(self, s, a, project_actions=True):
        """Transition function."""
        if project_actions:
            a = self.project_actions(a)
        next_true = self._step_two_state(s, a)
        next_true = next_true[0:2]
        next_true += self.noise * np.random.normal(loc=0., scale=1.,
                                                   size=next_true.shape)
        return next_true
        # next_image = self.render(next_true)
        # return (next_true, next_image)

    def is_fail(self, s):
        """Indicates whether the state results in failure."""
        # Unused.
        del s
        return False

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
        angle = np.random.uniform(self.angle_limits[0], self.angle_limits[1])
        angle_rate = np.random.uniform(self.angular_rate_limits[0],
                                       self.angular_rate_limits[1])
        true_state = np.array([angle, angle_rate])
        return true_state
        # return (true_state, self.render(true_state))

    def _step_two_state(self, s, a):
        """Computes the next state from the current state and the action."""
        torque_action = a

        # Add noise to the torque action.
        torque_action += np.random.uniform(-self.force_noise, self.force_noise)

        # Clip the action to valid values.
        torque_action = np.clip(torque_action, self.avail_force[0],
                                self.avail_force[1])

        # Add the action to the state so it can be passed to _dsdt.
        s_aug = np.append(s, torque_action)

        ## Compute next state.
        ns = solve_ivp(self._dsdt2, (0., self.dt), s_aug).y[0:2, -1]
        # Project variables to valid space.
        theta = wrap(ns[StateIndex.THETA], self.angle_limits[0],
                     self.angle_limits[1])
        ns[StateIndex.THETA] = bound(theta, self.angle_limits[0],
                                     self.angle_limits[1])
        ns[StateIndex.THETA_DOT] = bound(ns[StateIndex.THETA_DOT],
                                         self.angular_rate_limits[0],
                                         self.angular_rate_limits[1])

        return ns

    def _dsdt2(self, t, s_augmented):
        """Calculates derivatives at a given state."""
        # Unused.
        del t

        # Extracting current state and action.
        theta = s_augmented[StateIndex.THETA]
        theta_dot = s_augmented[StateIndex.THETA_DOT]
        torque = s_augmented[StateIndex.TORQUE]

        # Useful mid-calculation.
        sine = np.sin(np.pi - theta)

        # Theta is clockwise, starting from zero rad on top,
        # which is pi-\theta from tedrake's reference
        # Inertia constant: (1. / 3. * self.mass_pend * self.length**2).
        theta_double_dot_num = (self.mass_pend * self.accel_g * self.length * sine
                                - torque)
        theta_double_dot_denum = 1. / 3. * self.mass_pend * self.length**2
        theta_double_dot = theta_double_dot_num / theta_double_dot_denum

        # Derivatives.
        return np.array([theta_dot, theta_double_dot, 0.0])