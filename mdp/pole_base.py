import numpy as np
from numpy import pi
import os
from pathlib import Path

from mdp.common import StateIndex

root_path = str(Path(os.path.dirname(os.path.abspath(__file__))).parent)
os.sys.path.append(root_path)

class PoleBase(object):
    """
    The base class to store common attributes for pendulum and cartpole
    Basically, the MDP works as follows:
        - Define a function ds_dt, which computes the derivatives of state w.r.t time step t (from Tedrake's textbook)
        - The time interval between 2 consecutive time steps is determined by frequency, time_interval = 1./frequency
        - Use solve_ivp package to solve the differiential equation and compute the next state
    """
    # environment specifications
    earth_gravity = 9.81
    pend_mass = 0.1
    cart_mass = 1.0
    length = 0.5
    # reward if close to goal
    goal_reward = 1

    def __init__(self):
        assert np.all(2*pi / self.time_interval > np.abs(self.angular_velocity_range)), \
            """
            WARNING: Your step size is too small or the angular rate limit is too large.
            This could lead to a situation in which the pole is at the same state in 2
            consecutive time step (the pole finishes a round).
            """

    def take_step(self, s, u): # compute the next state given the current state and action
        pass

    def ds_dt(self, t, s): # derivative of s w.r.t t
        pass

    def transition_function(self, s, u): # compute next state and add noise
        s_next = self.take_step(s, u)
        # add noise
        s_next += self.noise * np.random.randn(*s_next.shape)
        return s_next

    def render(self, s):
        pass

    def is_goal(self, s):
        """Check if the pendulum is in goal region"""
        angle = s[StateIndex.THETA]
        return self.goal_range[0] <= angle <= self.goal_range[1]

    def is_fail(self, s):
        pass

    def reward_function(self, s):
        """Reward function."""
        return int(self.is_goal(s)) * self.goal_reward

    def sample_random_state(self):
        pass

    def sample_random_action(self):
        """Sample a random action from action range."""
        return np.array(
            [np.random.uniform(self.action_range[0], self.action_range[1])])

    def sample_extreme_action(self):
        """Sample a random extreme action from action range."""
        return np.array(
            [np.random.choice([self.action_range[0], self.action_range[1]])])