import numpy as np
from PIL import Image, ImageDraw

class PlanarObstaclesMDP(object):
    width = 40
    height = 40

    obstacles = np.array([[20.5, 5.5], [20.5, 13.5], [20.5, 27.5], [20.5, 35.5], [10.5, 20.5], [30.5, 20.5]])
    obstacles_r = 2.5 # radius of the obstacles when rendering
    half_agent_size = 1.5 # robot half-width

    position_range = np.array([half_agent_size, width - half_agent_size])

    action_dim = 2

    def __init__(self, rw_rendered=1, max_step=3,
                 goal=[37,37], goal_thres=2, noise = 0):
        self.rw_rendered = rw_rendered
        self.max_step = max_step
        self.action_range = np.array([-max_step, max_step])
        self.goal = goal
        self.goal_thres = goal_thres
        self.noise = noise
        super(PlanarObstaclesMDP, self).__init__()

    def is_valid_state(self, s):
        # check if the agent runs out of map
        if np.any(s < self.position_range[0]) or np.any(s > self.position_range[1]):
            return False

        # check if the agent crosses any obstacle (the obstacle is inside the agent)
        top, bot = s[0] - self.half_agent_size, s[0] + self.half_agent_size
        left, right = s[1] - self.half_agent_size, s[1] + self.half_agent_size
        for obs in self.obstacles:
            if top <= obs[0] <= bot and left <= obs[1] <= right:
                return False
        return True

    def take_step(self, s, u, anneal_ratio=0.9): # compute the next state given the current state and action
        u = np.clip(u, self.action_range[0], self.action_range[1])

        s_next = np.clip(s + u, self.position_range[0], self.position_range[1])
        if not self.is_valid_state(s_next):
            return s
        return s_next

    def transition_function(self, s, u): # compute next state and add noise
        s_next = self.take_step(s, u)
        # sample noise until get a valid next state
        sample_noise = self.noise * np.random.randn(*s_next.shape)
        return np.clip(s_next + sample_noise, self.position_range[0], self.position_range[1])

    def render(self, s):
        top, bottom, left, right = self.get_pixel_location(s)
        x = self.generate_env()
        x[top:bottom, left:right] = 1.  # robot is white on black background
        return x

    def get_pixel_location(self, s):
        # return the location of agent when rendered
        center_x, center_y = int(round(s[0])), int(round(s[1]))
        top = center_x - self.rw_rendered
        bottom = center_x + self.rw_rendered
        left = center_y - self.rw_rendered
        right = center_y + self.rw_rendered
        return top, bottom, left, right

    def generate_env(self):
        """
        return the image with 6 obstacles
        """
        img_arr = np.zeros(shape=(self.width, self.height))

        img_env = Image.fromarray(img_arr)
        draw = ImageDraw.Draw(img_env)
        for y, x in self.obstacles:
            draw.ellipse((int(x)-int(self.obstacles_r), int(y)-int(self.obstacles_r),
                        int(x)+int(self.obstacles_r), int(y)+int(self.obstacles_r)), fill=255)
        img_env = img_env.convert('L')

        img_arr = np.array(img_env) / 255.
        return img_arr

    def is_goal(self, s):
        return np.sqrt(np.sum((s - self.goal)**2)) <= self.goal_thres

    def is_fail(self, s):
        return False

    def reward_function(self, s):
        if self.is_goal(s):
            reward = 1
        else:
            reward = 0
        return reward

    def sample_random_state(self):
        while True:
            s = np.random.uniform(self.half_agent_size, self.width - self.half_agent_size, size = 2)
            if self.is_valid_state(s):
                return s

    def is_low_error(self, u, epsilon = 0.1):
        rounded_u = np.round(u)
        diff = np.abs(u - rounded_u)
        return np.all(diff <= epsilon)

    def is_valid_action(self, s, u):
        return self.is_low_error(u) and self.is_valid_state(s + u)

    def sample_valid_random_action(self, s):
        while True:
            u = np.random.uniform(self.action_range[0], self.action_range[1], size=self.action_dim)
            if self.is_valid_action(s, u):
                return u

    def sample_random_action(self):
        return np.random.uniform(self.action_range[0], self.action_range[1], size=self.action_dim)

    def sample_extreme_action(self):
        x_direction = np.random.choice([self.action_range[0], self.action_range[1]])
        y_direction = np.random.choice([self.action_range[0], self.action_range[1]])
        return np.array([x_direction, y_direction])
