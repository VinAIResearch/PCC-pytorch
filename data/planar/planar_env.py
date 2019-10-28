import numpy as np
from PIL import Image, ImageDraw

class PlanarEnv(object):
    width = 40
    height = 40
    obstacles_center = np.array([[20.5, 5.5], [20.5, 12.5], [20.5, 27.5], [20.5, 35.5], [10.5, 20.5], [30.5, 20.5]])

    r_overlap = 0.5 # agent cannot be in any rectangular area with obstacles as centers and half-width = 0.5
    r = 2.5 # radius of the obstacles when rendered
    rw = 3 # robot half-width
    rw_rendered = 2 # robot half-width when rendered
    max_step_len = 3

    def __init__(self):
        super(PlanarEnv, self).__init__()

    def random_state(self):
        s_x = np.random.uniform(low = self.rw, high = self.height - self.rw)
        s_y = np.random.uniform(low = self.rw, high = self.width - self.rw)
        s = np.array([s_x, s_y])
        return s

    def random_step(self, s):
        # draw a random step until it doesn't collidie with the obstacles
        while True:
            u = np.random.uniform(low = -self.max_step_len, high = self.max_step_len, size = 2)
            s_next = s + u
            if (not self.is_colliding(s_next) and self.is_valid(s, u, s_next)):
                return u, s_next

    def is_valid(self, s, u, s_next, epsilon = 0.1):
        # if the difference between the action and the actual distance between x and x_next are in range(0,epsilon)
        top, bottom, left, right = self.get_pixel_location(s)
        top_next, bottom_next, left_next, right_next = self.get_pixel_location(s_next)
        x_diff = np.array([top_next - top, left_next - left], dtype=np.float)
        return (not np.sqrt(np.sum((x_diff - u)**2)) > epsilon)

    def is_colliding(self, s):
        """
        :param s: the continuous coordinate (x, y) of the agent center
        :return: if agent body overlaps with obstacles
        """
        if np.any([s - self.rw < 0, s + self.rw > self.height]):
            return True
        x, y = s[0], s[1]
        for obs in self.obstacles_center:
            if np.abs(obs[0] - x) <= self.r_overlap and np.abs(obs[1] - y) <= self.r_overlap:
                return True
        return False

    def render_state(self, s):
        top, bottom, left, right = self.get_pixel_location(s)
        x = self.generate_env()
        x[top:bottom, left:right] = 1.  # robot is white on black background
        return x

    def generate_env(self):
        """
        return the environment with 6 obstacles
        """
        # print ('Making the environment...')
        img_arr = np.zeros(shape=(self.width, self.height))

        img_env = Image.fromarray(img_arr)
        draw = ImageDraw.Draw(img_env)
        for y, x in self.obstacles_center:
            draw.ellipse((int(x)-int(self.r), int(y)-int(self.r), int(x)+int(self.r), int(y)+int(self.r)), fill=255)
        img_env = img_env.convert('L')

        img_arr = np.array(img_env) / 255.
        return img_arr

    def get_pixel_location(self, s):
        # return the location of agent when rendered
        center_x, center_y = int(round(s[0])), int(round(s[1]))
        top = center_x - self.rw_rendered
        bottom = center_x + self.rw_rendered
        left = center_y - self.rw_rendered
        right = center_y + self.rw_rendered
        return top, bottom, left, right