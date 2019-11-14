import numpy as np
from numpy import pi
import os
from pathlib import Path

from mdp.common import StateIndex

root_path = str(Path(os.path.dirname(os.path.abspath(__file__))).parent)
os.sys.path.append(root_path)

# the base class to store common attributes for pendulum and cartpole
class PoleBase(object):
    # environment specifications
    earth_gravity = 9.81
    pend_mass = 0.1
    cart_mass = 1.0
    length = 0.5
    # reward if close to goal
    goal_reward = 1

    def __init__(self):
        assert self._is_params_valid()

    def _is_params_valid(self):
        """Check if the parameters are valid."""
        # time_interval: the time interval between 2 consecutive time steps
        if not ((2 * pi / self.time_interval > self.angular_velocity_range[1]) and
                (2 * pi / self.time_interval > -self.angular_velocity_range[0])):
            err_msg = """
            WARNING: Your step size is too small or the angular rate limit is too large.
            This could lead to a situation in which the pole is at the same state in 2
            consecutive time step (the pole finishes a round).
            """
            print(err_msg)
            return False

        return True

    def take_step(self, s, u):
        pass

    def ds_dt(self, t, s):
        pass

    def transition_function(self, s, u):
        """Transition function."""
        s_next = self.take_step(s, u) # last index is the action applied
        # add noise
        s_next += self.noise * np.random.rand(*s_next.shape)
        return s_next

    def render(self, s):
        pass

    def is_goal(self, s):
        """Check if the pendulum is in goal region"""
        angle = s[StateIndex.THETA]
        return self.goal_range[0] <= angle <= self.goal_range[1]

    def reward_function(self, s):
        """Reward function."""
        return int(self.is_goal(s)) * self.goal_reward

    def sample_random_state(self):
        pass

    # def project_actions(self, a_seq): # clipping a sequence of actions
    #     """Project actions onto actions space."""
    #     # Capping on minimum threshold.
    #     a_seq[a_seq < self.avail_force[0]] = self.avail_force[0]
    #     # Capping on maximum threshold.
    #     a_seq[a_seq > self.avail_force[1]] = self.avail_force[1]
    #     return a_seq

    def sample_random_action(self):
        """Sample a random action from action range."""
        return np.atleast_1d(
            np.random.uniform(self.action_range[0], self.action_range[1]))

    def sample_extreme_action(self):
        """Sample a random extreme action from action range."""
        return np.atleast_1d(
            np.random.choice([self.action_range[0], self.action_range[1]]))