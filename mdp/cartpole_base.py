"""Cartpole base MDP."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from matplotlib.mlab import rk4
# from scipy.integrate import ode as rk4
import numpy as np
from numpy import pi
from scipy.integrate import solve_ivp
import os
from pathlib import Path

root_path = str(Path(os.path.dirname(os.path.abspath(__file__))).parent)
os.sys.path.append(root_path)

from mdp.common import bound
from mdp.common import StateIndex
from mdp.common import wrap


class CartPoleBase(object):
    """Base class for coretpole and pole."""
    ## Constants.
    # Gravitational acceleration.
    accel_g = 9.81

    # Pendulum mass.
    mass_pend = 0.1
    # Cart mass.
    mass_cart = 1.0
    # Pole length.
    length = 0.5

    # # Integration type. Currently supports 'solve_ivp' and 'rk4' (default).
    # int_type = 'rk4'
    # Number of steps in numerical integration ('rk4')
    integration_steps = 10
    # Integration step size.
    dt = 0.02

    # Force boundaries.
    avail_force = np.array([-10.0, 10.0])
    # Force noise std for actions.
    force_noise = 0.

    # Magnitude of additive noise to the transitions.
    noise = 0.

    #: Reward received on each step the pendulum is in the goal region.
    goal_reward = 1

    def __init__(self):
        assert self._is_params_valid()

    def _is_params_valid(self):
        """Check if the parameters are valid."""
        if not ((2 * pi / self.dt > self.angular_rate_limits[1]) and
                (2 * pi / self.dt > -self.angular_rate_limits[0])):
            err_str = """
            WARNING: If the bound on angular velocity is large compared with
            the time discretization, seemingly 'optimal' performance
            might result from a stroboscopic-like effect.
            For example, if dt = 1.0 sec, and the angular rate limits
            exceed -2pi or 2pi respectively, then it is possible that
            between consecutive timesteps, the pendulum will have
            the same position, even though it really completed a
            rotation, and thus we will find a solution that commands
            the pendulum to spin with angular rate in multiples of
            2pi / dt instead of truly remaining stationary (0 rad/s) at goal.
            """
            print(err_str)
            print('Your selection, dt=', self.dt, 'and limits',
                  self.angular_rate_limits, 'Are at risk.')
            print(
                'Reduce your timestep dt (to increase # timesteps) or reduce angular'
                ' rate limits so that 2pi / dt > max(AngularRateLimit)')
            print('Currently, 2pi / dt = ', 2 * pi / self.dt,
                  ', angular rate limits shown above.')
            return False

        return True

    def render(self, s):
        """Renders the image."""
        pass

    def transition_function(self, s, a):
        """Transition function."""
        pass

    def is_fail(self, s):
        """Indicates whether the state results in failure."""
        pass

    def is_goal(self, s):
        """Inidicates whether the state achieves the goal."""
        pass

    def reward_function(self, s):
        """Reward function."""
        pass

    def sample_random_state(self):
        """Sample a random state."""
        pass

    def project_actions(self, a_seq): # clipping a sequence of actions
        """Project actions onto actions space."""
        # Capping on minimum threshold.
        a_seq[a_seq < self.avail_force[0]] = self.avail_force[0]
        # Capping on maximum threshold.
        a_seq[a_seq > self.avail_force[1]] = self.avail_force[1]
        return a_seq

    def sample_random_action(self):
        """Sample a random action from available force."""
        return np.atleast_1d(
            np.random.uniform(self.avail_force[0], self.avail_force[1]))

    def sample_extreme_action(self):
        """Sample a random extreme action from available force."""
        return np.atleast_1d(
            np.random.choice([self.avail_force[0], self.avail_force[1]]))

    def _step_four_state(self, s, a):
        """Computes the next state from the current state and the action."""
        force_action = a

        # Add noise to the force action.
        force_action += np.random.uniform(-self.force_noise, self.force_noise)

        # Clip the action to valid values.
        force_action = np.clip(force_action, self.avail_force[0],
                               self.avail_force[1])

        # Add the action to the state so it can be passed to _dsdt.
        s_aug = np.append(s, force_action)

        ## Compute next state.
        # Type of integration and integration step.
        ns = solve_ivp(self._dsdt, (0., self.dt), s_aug).y[0:4, -1]

        # Project variables to valid space.
        theta = wrap(ns[StateIndex.THETA], self.angle_limits[0],
                     self.angle_limits[1])
        ns[StateIndex.THETA] = bound(theta, self.angle_limits[0],
                                     self.angle_limits[1])  # for what ??? min < theta < max already
        ns[StateIndex.THETA_DOT] = bound(ns[StateIndex.THETA_DOT],
                                         self.angular_rate_limits[0],
                                         self.angular_rate_limits[1])
        ns[StateIndex.X_DOT] = bound(ns[StateIndex.X_DOT], self.velocity_limits[0],
                                     self.velocity_limits[1])

        return ns

    def _dsdt(self, t, s_augmented):
        """Calculates derivatives at a given state."""
        # Unused.
        del t

        mass_combined = self.mass_cart + self.mass_pend

        # Extracting current state and action.
        theta = s_augmented[StateIndex.THETA]
        theta_dot = s_augmented[StateIndex.THETA_DOT]
        x_dot = s_augmented[StateIndex.X_DOT]
        force = s_augmented[StateIndex.FORCE]

        # Useful mid-calculations.
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        calc_help = force + self.mass_pend * self.length * theta_dot**2 * sin_theta

        # Theta double-dot.
        theta_double_dot_num = (mass_combined * self.accel_g * sin_theta
                                - cos_theta * calc_help)
        theta_double_dot_denum = (4. / 3 * mass_combined * self.length
                                  - self.mass_pend * self.length * cos_theta**2)
        theta_double_dot = theta_double_dot_num / theta_double_dot_denum

        # X double-dot.
        x_double_dot_num = (calc_help - self.mass_pend * self.length
                            * theta_double_dot * cos_theta)
        x_double_dot = x_double_dot_num / mass_combined

        # Derivatives.
        return np.array([theta_dot, theta_double_dot, x_dot, x_double_dot, 0.0])