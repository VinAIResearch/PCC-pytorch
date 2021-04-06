import numpy as np
from mdp.common import StateIndex, wrap
from mdp.pole_base import PoleBase
from PIL import Image, ImageDraw
from scipy.integrate import solve_ivp


class ThreePoleMDP(PoleBase):
    # goal_range
    goal_range = [-np.pi / 6, np.pi / 6]

    # state range
    angle_1_range = [-np.pi, np.pi]
    angle_2_range = [-2.0 * np.pi / 3, 2.0 * np.pi / 3]
    angle_3_range = [-np.pi / 3.0, np.pi / 3.0]
    angular_velocity_range = [-0.5 * np.pi, 0.5 * np.pi]

    mass_pend_1 = 0.1
    mass_pend_2 = 0.1
    mass_pend_3 = 0.1
    length_1 = 0.5
    length_2 = 0.5
    length_3 = 0.5

    action_dim = 3

    def __init__(self, width=80, height=80, frequency=50, noise=0.0, torque=1.0, line_width=5):
        """
        Args:
          width: width of the rendered image.
          height: height of the rendered image.
          frequency: the simulator frequency, i.e., the number of steps in 1 second.
          noise: noise level
          torque: the maximal torque which can be applied
          render_width: width of the line in the rendered image.
        """
        self.width = width
        self.height = height
        self.time_interval = 1.0 / frequency
        self.noise = noise
        self.action_range = np.array([-torque, torque])

        self.line_width = line_width
        self.visual_length = (width / 6) - 2

        super(ThreePoleMDP, self).__init__()

    def take_step(self, s, a):
        """Computes the next state from the current state and the action."""
        torque_1_action = a[0]
        # Clip the action to valid values.
        torque_1_action = np.clip(torque_1_action, self.action_range[0], self.action_range[1])

        torque_2_action = a[1]
        # Clip the action to valid values.
        torque_2_action = np.clip(torque_2_action, self.action_range[0], self.action_range[1])

        torque_3_action = a[2]
        # Clip the action to valid values.
        torque_3_action = np.clip(torque_3_action, self.action_range[0], self.action_range[1])

        # Add the action to the state so it can be passed to _dsdt.
        s_aug = np.append(s, np.array([torque_1_action, torque_2_action, torque_3_action]))

        # Compute next state.
        # Type of integration and integration step.
        dt_in = self.time_interval
        if (
            self.goal_range[0] < s[StateIndex.THETA_1] < self.goal_range[1]
            and self.goal_range[0] < s[StateIndex.THETA_1] + s[StateIndex.THETA_2] < self.goal_range[1]
            and self.goal_range[0]
            < s[StateIndex.THETA_1] + s[StateIndex.THETA_2] + s[StateIndex.THETA_3]
            < self.goal_range[1]
        ):
            dt_in = self.time_interval
        else:
            dt_in = self.time_interval * 2.5

        ns = solve_ivp(self.ds_dt, (0.0, dt_in), s_aug).y[0:6, -1]

        # Project variables to valid space.
        theta_1 = wrap(ns[StateIndex.THETA_1], self.angle_1_range[0], self.angle_1_range[1])
        ns[StateIndex.THETA_1] = np.clip(theta_1, self.angle_1_range[0], self.angle_1_range[1])
        ns[StateIndex.THETA_1_DOT] = np.clip(
            ns[StateIndex.THETA_1_DOT], self.angular_velocity_range[0], self.angular_velocity_range[1]
        )

        theta_2 = wrap(ns[StateIndex.THETA_2], self.angle_2_range[0], self.angle_2_range[1])
        ns[StateIndex.THETA_2] = np.clip(theta_2, self.angle_2_range[0], self.angle_2_range[1])
        ns[StateIndex.THETA_2_DOT] = np.clip(
            ns[StateIndex.THETA_2_DOT], self.angular_velocity_range[0], self.angular_velocity_range[1]
        )

        theta_3 = wrap(ns[StateIndex.THETA_3], self.angle_3_range[0], self.angle_3_range[1])
        ns[StateIndex.THETA_3] = np.clip(theta_3, self.angle_3_range[0], self.angle_3_range[1])
        ns[StateIndex.THETA_3_DOT] = np.clip(
            ns[StateIndex.THETA_3_DOT], self.angular_velocity_range[0], self.angular_velocity_range[1]
        )

        return ns

    def ds_dt(self, t, s_augmented):
        """Calculates derivatives at a given state."""
        # Unused.
        del t

        # Extracting current state and action.
        theta_1 = s_augmented[StateIndex.THETA_1]
        theta_1_dot = s_augmented[StateIndex.THETA_1_DOT]
        theta_2 = s_augmented[StateIndex.THETA_2]
        theta_2_dot = s_augmented[StateIndex.THETA_2_DOT]
        theta_3 = s_augmented[StateIndex.THETA_3]
        theta_3_dot = s_augmented[StateIndex.THETA_3_DOT]

        theta_dot = np.array([theta_1_dot, theta_2_dot, theta_3_dot])

        torque_1 = s_augmented[StateIndex.TORQUE_3_1]
        torque_2 = s_augmented[StateIndex.TORQUE_3_2]
        torque_3 = s_augmented[StateIndex.TORQUE_3_3]
        torque = np.array([torque_1, torque_2, torque_3])

        # Useful mid-calculation.
        # NOTE: the angle here is clock-wise
        # which is -\theta from tedrake's reference

        sine_1 = np.sin(-theta_1)
        sine_2 = np.sin(-theta_2)
        sine_3 = np.sin(-theta_3)
        sine_2_3 = np.sin(-(theta_2 + theta_3))

        # cosine_1 = np.cos(np.pi - theta_1)
        cosine_2 = np.cos(-theta_2)
        cosine_3 = np.cos(-theta_3)
        cosine_2_3 = np.cos(-(theta_2 + theta_3))

        sine_1_2 = np.sin(-(theta_1 + theta_2))
        # cosine_1_2 = np.cos(np.pi - (theta_1 + theta_2))
        sine_1_2_3 = np.sin(-(theta_1 + theta_2 + theta_3))

        i_1 = 1.0 / 3.0 * self.mass_pend_1 * self.length_1 ** 2
        i_2 = 1.0 / 3.0 * self.mass_pend_2 * self.length_2 ** 2
        i_3 = 1.0 / 3.0 * self.mass_pend_3 * self.length_3 ** 2

        length_c1 = self.length_1 / 2.0
        length_c2 = self.length_2 / 2.0
        length_c3 = self.length_3 / 2.0

        # point mass version, not a rod, so no inertia and no center-of-mass
        alpha_1 = i_1 + (self.mass_pend_2 + self.mass_pend_3) * self.length_1 ** 2
        alpha_2 = i_2 + self.mass_pend_3 * self.length_2 ** 2
        alpha_3 = (self.mass_pend_2 * length_c2 + self.mass_pend_3 * self.length_2) * self.length_1
        alpha_4 = i_3
        alpha_5 = self.mass_pend_3 * self.length_1 * length_c3
        alpha_6 = self.mass_pend_3 * self.length_2 * length_c3

        h_11 = alpha_1 + alpha_2 + alpha_4 + 2 * alpha_5 * cosine_2_3 + 2 * alpha_3 * cosine_2 + 2 * alpha_6 * cosine_3
        h_12 = alpha_2 + alpha_4 + alpha_3 * cosine_2 + alpha_5 * cosine_2_3 + 2 * alpha_6 * cosine_3
        h_13 = alpha_4 + alpha_5 * cosine_2_3 + alpha_6 * cosine_3

        h_21 = h_12
        h_22 = alpha_2 + alpha_4 + 2 * alpha_6 * cosine_3
        h_23 = alpha_4 + alpha_6 * cosine_3

        h_31 = h_13
        h_32 = h_23
        h_33 = alpha_4
        h_mat = np.array([[h_11, h_12, h_13], [h_21, h_22, h_23], [h_31, h_32, h_33]])

        beta_1 = (
            self.mass_pend_1 * length_c1 + self.mass_pend_2 * self.length_1 + self.mass_pend_3 * self.length_1
        ) * self.earth_gravity
        beta_2 = (self.mass_pend_2 * length_c2 + self.mass_pend_3 * self.length_2) * self.earth_gravity
        beta_3 = self.mass_pend_3 * self.earth_gravity * length_c3

        c_11 = (
            alpha_5 * (theta_2_dot + theta_3_dot) * sine_2_3
            + alpha_3 * theta_2_dot * sine_2
            + alpha_6 * theta_3_dot * sine_3
        )
        c_12 = (
            alpha_5 * (theta_1_dot + theta_2_dot + theta_3_dot) * sine_2_3
            + alpha_3 * (theta_1_dot + theta_2_dot) * sine_2
            + alpha_6 * theta_3_dot * sine_3
        )
        c_13 = (theta_1_dot + theta_2_dot + theta_3_dot) * (alpha_5 * sine_2_3 + alpha_6 * sine_3)
        c_21 = -alpha_5 * theta_1_dot * sine_2_3 - alpha_3 * theta_1_dot * sine_2 + alpha_6 * theta_3_dot * sine_3
        c_22 = alpha_6 * theta_3_dot * sine_3
        c_23 = alpha_6 * (theta_1_dot + theta_2_dot + theta_3_dot) * sine_3
        c_31 = -alpha_5 * theta_1_dot * sine_2_3 - alpha_6 * (theta_1_dot + theta_2_dot) * sine_3
        c_32 = -alpha_6 * (theta_1_dot + theta_2_dot) * sine_3
        c_33 = 0.0
        c_mat = np.array([[c_11, c_12, c_13], [c_21, c_22, c_23], [c_31, c_32, c_33]])

        g_1 = -beta_1 * sine_1 - beta_2 * sine_1_2 - beta_3 * sine_1_2_3
        g_2 = -beta_2 * sine_1_2 - beta_3 * sine_1_2_3
        g_3 = -beta_3 * sine_1_2_3
        g_mat = np.array([g_1, g_2, g_3])

        b_mat = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        theta_double_dot = -np.linalg.pinv(h_mat + 1e-6 * np.eye(len(h_mat))).dot(
            b_mat.dot(torque) - c_mat.dot(-theta_dot) - g_mat
        )

        theta_1_double_dot = theta_double_dot[0]
        theta_2_double_dot = theta_double_dot[1]
        theta_3_double_dot = theta_double_dot[2]

        # Derivatives.
        return np.array(
            [
                theta_1_dot,
                theta_1_double_dot,
                theta_2_dot,
                theta_2_double_dot,
                theta_3_dot,
                theta_3_double_dot,
                0.0,
                0.0,
                0.0,
            ]
        )

    def render(self, s):
        im = Image.new("L", (self.width, self.height))
        draw = ImageDraw.Draw(im)
        # Draw background.
        draw.rectangle((0, 0, self.width, self.height), fill=0)

        # Pole 1 location.
        xstart_1 = im.size[0] / 2.0
        ystart_1 = im.size[1] / 2.0
        xend_1 = xstart_1 + np.sin(s[0]) * self.visual_length
        yend_1 = ystart_1 - np.cos(s[0]) * self.visual_length

        # Draw pole 1.
        draw.line((xstart_1, ystart_1, xend_1, yend_1), width=self.line_width, fill=255)

        # Pole 2 location.
        xstart_2 = xend_1
        ystart_2 = yend_1
        xend_2 = xstart_2 + np.sin(s[0] + s[2]) * self.visual_length
        yend_2 = ystart_2 - np.cos(s[0] + s[2]) * self.visual_length

        # Draw pole 2.
        draw.line((xstart_2, ystart_2, xend_2, yend_2), width=self.line_width, fill=255)

        # Pole 2 location.
        xstart_3 = xend_2
        ystart_3 = yend_2
        xend_3 = xstart_3 + np.sin(s[0] + s[2] + s[4]) * self.visual_length
        yend_3 = ystart_3 - np.cos(s[0] + s[2] + s[4]) * self.visual_length

        # Draw pole 2.
        draw.line((xstart_3, ystart_3, xend_3, yend_3), width=self.line_width, fill=255)

        return np.expand_dims(np.asarray(im) / 255.0, axis=-1)

    def is_fail(self, s):
        """Indicates whether the state results in failure."""
        # Unused.
        del s
        return False

    def is_goal(self, s):
        """Inidicates whether the state achieves the goal."""
        angle_1 = s[StateIndex.THETA_1]
        angle_2 = s[StateIndex.THETA_2]
        angle_3 = s[StateIndex.THETA_3]
        if (
            self.goal_range[0] < angle_1 < self.goal_range[1]
            and self.goal_range[0] < angle_1 + angle_2 < self.goal_range[1]
            and self.goal_range[0] < angle_1 + angle_2 + angle_3 < self.goal_range[1]
        ):
            return True
        else:
            return False

    def sample_random_action(self):
        """Sample a random action from available force."""
        return np.atleast_1d(np.random.uniform(self.action_range[0], self.action_range[1], self.action_dim))

    def sample_extreme_action(self):
        """Sample a random extreme action from available force."""
        return np.atleast_1d(np.random.choice([self.action_range[0], self.action_range[1]], self.action_dim))

    def sample_random_state(self):
        """Sample a random state."""
        angle_1 = np.random.uniform(self.angle_1_range[0], self.angle_1_range[1])
        angle_1_rate = np.random.uniform(self.angular_velocity_range[0], self.angular_velocity_range[1])
        angle_2 = np.random.uniform(self.angle_2_range[0], self.angle_2_range[1])
        angle_2_rate = np.random.uniform(self.angular_velocity_range[0], self.angular_velocity_range[1])
        angle_3 = np.random.uniform(self.angle_3_range[0], self.angle_3_range[1])
        angle_3_rate = np.random.uniform(self.angular_velocity_range[0], self.angular_velocity_range[1])
        true_state = np.array([angle_1, angle_1_rate, angle_2, angle_2_rate, angle_3, angle_3_rate])
        return true_state
