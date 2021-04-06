import numpy as np
from mdp.common import StateIndex, wrap
from mdp.pole_base import PoleBase
from PIL import Image, ImageDraw
from scipy.integrate import solve_ivp


class PendulumMDP(PoleBase):
    # goal range
    goal_range = [-np.pi / 6, np.pi / 6]

    # state range
    angle_range = [-np.pi, np.pi]
    angular_velocity_range = [-3 * np.pi, 3 * np.pi]

    action_dim = 1

    def __init__(self, width=48, height=48, frequency=50, noise=0.0, torque=1.0, render_width=4):
        """
        Args:
          width: width of the rendered image.
          height: height of the rendered image.
          frequency: the simulator frequency, i.e., the number of steps in 1 second.
          noise: noise level
          torque: the maximal torque which can be applied
          render_width: width of the pendulum in the rendered image.
        """
        self.width = width
        self.height = height
        self.time_interval = 1.0 / frequency
        self.noise = noise
        self.action_range = np.array([-torque, torque])

        self.render_width = render_width
        self.render_length = (width / 2) - 2

        super(PendulumMDP, self).__init__()

    def take_step(self, s, u):
        # clip the action
        u = np.clip(u, self.action_range[0], self.action_range[1])

        # concatenate s and u to pass through ds_dt
        s_aug = np.append(s, u)

        # solve the differientable equation to compute next state
        s_next = solve_ivp(self.ds_dt, (0.0, self.time_interval), s_aug).y[0:2, -1]  # last index is the action applied

        # project state to the valid range
        s_next[StateIndex.THETA] = wrap(s_next[StateIndex.THETA], self.angle_range[0], self.angle_range[1])
        s_next[StateIndex.THETA_DOT] = np.clip(
            s_next[StateIndex.THETA_DOT], self.angular_velocity_range[0], self.angular_velocity_range[1]
        )

        return s_next

    def ds_dt(self, t, s_augmented):
        theta = s_augmented[StateIndex.THETA]
        theta_dot = s_augmented[StateIndex.THETA_DOT]
        torque = s_augmented[StateIndex.PEND_ACTION]

        # theta is w.r.t the upside vertical position, which is = pi - theta in tedrake's note
        sine = np.sin(np.pi - theta)
        theta_prime_num = self.pend_mass * self.earth_gravity * self.length * sine - torque
        theta_prime_denum = 1.0 / 3.0 * self.pend_mass * self.length ** 2  # the moment of inertia
        theta_double_dot = theta_prime_num / theta_prime_denum

        return np.array([theta_dot, theta_double_dot, 0.0])

    def render(self, s):
        im = Image.new("L", (self.width, self.height))
        draw = ImageDraw.Draw(im)
        # black background
        draw.rectangle((0, 0, self.width, self.height), fill=0)

        # pendulum location.
        x_center = im.size[0] / 2.0
        y_center = im.size[1] / 2.0
        x_end = x_center + np.sin(s[0]) * self.render_length
        y_end = y_center - np.cos(s[0]) * self.render_length

        # white pendulum
        draw.line((x_center, y_center, x_end, y_end), width=self.render_width, fill=255)

        return np.expand_dims(np.asarray(im) / 255.0, axis=-1)

    def is_fail(self, s):
        return False

    def sample_random_state(self):
        angle = np.random.uniform(self.angle_range[0], self.angle_range[1])
        angle_rate = np.random.uniform(self.angular_velocity_range[0], self.angular_velocity_range[1])
        true_state = np.array([angle, angle_rate])
        return true_state
