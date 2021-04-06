import numpy as np
from mdp.common import StateIndex, wrap
from mdp.pole_base import PoleBase
from PIL import Image, ImageDraw
from scipy.integrate import solve_ivp


class CartPoleMDP(PoleBase):
    goal_range = [-np.pi / 10, np.pi / 10]

    # state range
    angle_range = [-np.pi, np.pi]
    angular_velocity_range = [-2 * np.pi, 2 * np.pi]
    position_range = [-2.4, 2.4]
    velocity_range = [-6.0, 6.0]

    # sampling range
    angle_samp_range = 2 * goal_range

    # action range
    action_dim = 1
    action_range = np.array([-10.0, 10.0])

    def __init__(self, width=80, height=80, frequency=50, noise=0.0, render_width=6):
        """
        Args:
          width: width of the rendered image.
          height: height of the rendered image.
          frequency: the simulator frequency, i.e., the number of steps in 1 second.
          noise: noise level
          render_width: width of the pole in the rendered image.
        """
        self.width = width
        self.height = height
        self.time_interval = 1 / frequency
        self.noise = noise

        self.render_width = render_width
        self.render_length = height * 0.65
        self.cart_render_size = (width / 10.0, height / 20.0)

        super(CartPoleMDP, self).__init__()

    def take_step(self, s, u):
        # clip the action
        u = np.clip(u, self.action_range[0], self.action_range[1])

        # concatenate s and u to pass through ds_dt
        s_aug = np.append(s, u)

        # solve the differientable equation to compute next state
        s_next = solve_ivp(self.ds_dt, (0.0, self.time_interval), s_aug).y[0:4, -1]  # last index is the action applied

        # project state to the valid space.
        s_next[StateIndex.THETA] = wrap(s_next[StateIndex.THETA], self.angle_range[0], self.angle_range[1])
        s_next[StateIndex.THETA_DOT] = np.clip(
            s_next[StateIndex.THETA_DOT], self.angular_velocity_range[0], self.angular_velocity_range[1]
        )
        s_next[StateIndex.X_DOT] = np.clip(s_next[StateIndex.X_DOT], self.velocity_range[0], self.velocity_range[1])

        return s_next

    def ds_dt(self, t, s_augmented):
        mass_combined = self.cart_mass + self.pend_mass

        theta = s_augmented[StateIndex.THETA]
        theta_dot = s_augmented[StateIndex.THETA_DOT]
        x_dot = s_augmented[StateIndex.X_DOT]
        force = s_augmented[StateIndex.CARTPOLE_ACTION]

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        calc_help = force + self.pend_mass * self.length * theta_dot ** 2 * sin_theta

        # derivative of theta_dot
        theta_double_dot_num = mass_combined * self.earth_gravity * sin_theta - cos_theta * calc_help
        theta_double_dot_denum = 4.0 / 3 * mass_combined * self.length - self.pend_mass * self.length * cos_theta ** 2
        theta_double_dot = theta_double_dot_num / theta_double_dot_denum

        # derivative of x_dot
        x_double_dot_num = calc_help - self.pend_mass * self.length * theta_double_dot * cos_theta
        x_double_dot = x_double_dot_num / mass_combined

        return np.array([theta_dot, theta_double_dot, x_dot, x_double_dot, 0.0])

    def render(self, s):
        # black background.
        im = Image.new("L", (self.width, self.height))
        draw = ImageDraw.Draw(im)

        draw.rectangle((0, 0, self.width, self.height), fill=0)

        # cart location.
        x_center_image = im.size[0] / 2.0
        y_center_cart = im.size[1] - 2 * self.cart_render_size[1] - 2
        x_center_cart = x_center_image + (s[StateIndex.X] / self.position_range[1]) * (
            self.width / 2.0 - 1.0 * self.cart_render_size[0]
        )

        # pole location.
        x_pole_end = x_center_cart + np.sin([s[StateIndex.THETA]]) * self.render_length
        y_pole_end = y_center_cart - np.cos([s[StateIndex.THETA]]) * self.render_length

        # draw cart.
        draw.rectangle(
            (
                x_center_cart - self.cart_render_size[0],
                y_center_cart - self.cart_render_size[1],
                x_center_cart + self.cart_render_size[0],
                y_center_cart + self.cart_render_size[1],
            ),
            fill=255,
        )

        # draw pole.
        draw.line((x_center_cart, y_center_cart, x_pole_end, y_pole_end), width=self.render_width, fill=255)

        return np.expand_dims(np.asarray(im) / 255.0, axis=-1)

    def is_fail(self, s):
        """check if the current state is failed"""
        angle = s[StateIndex.THETA]
        position = s[StateIndex.X]
        return not (
            (self.goal_range[0] < angle < self.goal_range[1])
            and (self.position_range[0] < position < self.position_range[1])
        )

    def sample_random_state(self):
        angle = np.random.uniform(self.angle_samp_range[0], self.angle_samp_range[1])
        angle_rate = np.random.uniform(self.angular_velocity_range[0], self.angular_velocity_range[1])
        pos = np.random.uniform(self.position_range[0], self.position_range[1])
        vel = np.random.uniform(self.velocity_range[0], self.velocity_range[1])
        true_state = np.array([angle, angle_rate, pos, vel])
        return true_state
