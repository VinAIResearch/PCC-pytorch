import numpy as np
from PIL import Image, ImageDraw

class PlanarObstaclesMDP(object):
    width = 40
    height = 40
    obstacles = np.array([[20.5, 5.5], [20.5, 13.5], [20.5, 27.5], [20.5, 35.5], [10.5, 20.5], [30.5, 20.5]])
    obstacles_r = 2.5 # radius of the obstacles when rendered
    half_agent_size = 1.5 # robot half-width
    rw_rendered = 1 # robot half-width when rendered
    max_step = 3
    action_dim = 2

    def __init__(self, sampling = False, noise = 0):
        self.is_sampling = sampling
        self.noise = noise
        super(PlanarObstaclesMDP, self).__init__()

    def is_valid_state(self, s):
        if np.any(s < self.half_agent_size) or np.any(s > self.width - self.half_agent_size):
            return False
        # check if the agent crosses any obstacle
        obstacles = self.obstacles.astype(np.int)
        for obs in obstacles:
            dis = np.abs(obs - s)
            if np.all(dis <= 1):
                return False
        return True
        # check if the agent center is inside any obstacle
        # dis_to_obs = np.sqrt(np.sum((self.obstacles - s)**2, axis=1))
        # if np.any(dis_to_obs < self.obstacles_r):
        #     return False
        # return True

    def is_low_error(self, s, u, epsilon = 0.1):
        s_next = s + u
        # if the difference between the action and the actual distance between x and x_next are in range(0,epsilon)
        top, bottom, left, right = self.get_pixel_location(s)
        top_next, bottom_next, left_next, right_next = self.get_pixel_location(s_next)
        x_diff = np.array([top_next - top, left_next - left], dtype=np.float)
        return (not np.sqrt(np.sum((x_diff - u)**2)) > epsilon)
        # return True

    def is_valid_action(self, s, u):
        if self.is_sampling:
            if not self.is_low_error(s, u):
                return False
        s_next = s + u + self.noise * np.random.randn()
        if not self.is_valid_state(s_next):
            return False
        return True

    def sample_valid_random_state(self):
        while True:
            s = np.random.uniform(self.half_agent_size, self.width - self.half_agent_size, size = 2)
            if self.is_valid_state(s):
                return s

    def sample_valid_random_action(self, s):
        while True:
            u = np.random.uniform(-self.max_step, self.max_step, size=self.action_dim)
            if self.is_valid_action(s, u):
                return u

    def transition_function(self, s, u):
        u = np.clip(u, -self.max_step, self.max_step)
        if not self.is_valid_action(s, u):
            return s
        s_next = s + u + self.noise * np.random.randn()
        return s_next

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
        # topleft_x, topleft_y = int(s[0]), int(s[1])
        # top = topleft_x
        # bottom = topleft_x + self.rw_rendered + 1
        # left = topleft_y
        # right = topleft_y + self.rw_rendered + 1
        # return top, bottom, left, right

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