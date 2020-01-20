import os
from os import path
import numpy as np
from torch.utils.data import Dataset
import torch

from data import sample_planar
from data import sample_pole

torch.set_default_dtype(torch.float64)

class BaseDataset(Dataset):
    def __init__(self, data_path, sample_size, noise):
        self.sample_size = sample_size
        self.noise = noise
        self.data_path = data_path
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        self._process()
        self.data_x, self.data_u, self.data_x_next = torch.load(self.data_path + '{:d}_{:.0f}.pt'.format(self.sample_size, self.noise))

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, index):
        return self.data_x[index], self.data_u[index], self.data_x_next[index]

    def _process_image(self, img):
        pass

    def check_exists(self):
        return (path.exists(self.data_path + '{:d}_{:.0f}.pt'.format(self.sample_size, self.noise)))

    def _process(self):
        pass

class PlanarDataset(BaseDataset):
    width = 40
    height = 40
    action_dim = 2

    def __init__(self, sample_size, noise):
        data_path = 'data/planar/'
        super(PlanarDataset, self).__init__(data_path, sample_size, noise)

    def _process_image(self, img):
        return torch.from_numpy(img.flatten()).unsqueeze(0)

    def _process(self):
        if self.check_exists():
            return
        else:
            x_numpy_data, u_numpy_data, x_next_numpy_data, state_numpy_data, state_next_numpy_data = \
                                sample_planar.sample(sample_size=self.sample_size, noise=self.noise)
            data_len = len(x_numpy_data)

            # place holder for data
            data_x = torch.zeros(data_len, self.width * self.height)
            data_u = torch.zeros(data_len, self.action_dim)
            data_x_next = torch.zeros(data_len, self.width * self.height)

            for i in range(data_len):
                data_x[i] = self._process_image(x_numpy_data[i])
                data_u[i] = torch.from_numpy(u_numpy_data[i])
                data_x_next[i] = self._process_image(x_next_numpy_data[i])

            data_set = (data_x, data_u, data_x_next)

            with open(self.data_path + '{:d}_{:.0f}.pt'.format(self.sample_size, self.noise), 'wb') as f:
                torch.save(data_set, f)

class PendulumDataset(BaseDataset):
    width = 48
    height = 48 * 2
    action_dim = 1

    def __init__(self, sample_size, noise):
        data_path = 'data/pendulum/'
        super(PendulumDataset, self).__init__(data_path, sample_size, noise)

    def _process_image(self, img):
        x = np.vstack((img[:, :, 0], img[:, :, 1])).flatten()
        return torch.from_numpy(x).unsqueeze(0)

    def _process(self):
        if self.check_exists():
            return
        else:
            x_numpy_data, u_numpy_data, x_next_numpy_data, state_numpy_data, state_next_numpy_data = \
                sample_pole.sample(env_name='pendulum', sample_size=self.sample_size, noise=self.noise)
            data_len = len(x_numpy_data)

            # place holder for data
            data_x = torch.zeros(data_len, self.width * self.height)
            data_u = torch.zeros(data_len, self.action_dim)
            data_x_next = torch.zeros(data_len, self.width * self.height)

            for i in range(data_len):
                data_x[i] = self._process_image(x_numpy_data[i])
                data_u[i] = torch.from_numpy(u_numpy_data[i])
                data_x_next[i] = self._process_image(x_next_numpy_data[i])

            data_set = (data_x, data_u, data_x_next)

            with open(self.data_path + '{:d}_{:.0f}.pt'.format(self.sample_size, self.noise), 'wb') as f:
                torch.save(data_set, f)

class CartPoleDataset(BaseDataset):
    width = 80
    height = 80 * 2
    action_dim = 1

    def __init__(self, sample_size, noise):
        data_path = 'data/cartpole/'
        super(CartPoleDataset, self).__init__(data_path, sample_size, noise)

    def _process_image(self, img):
        x = torch.zeros(size=(2, self.width, self.width))
        x[0, :, :] = torch.from_numpy(img[:, :, 0])
        x[1, :, :] = torch.from_numpy(img[:, :, 1])
        return x.unsqueeze(0)

    def _process(self):
        if self.check_exists():
            return
        else:
            x_numpy_data, u_numpy_data, x_next_numpy_data, state_numpy_data, state_next_numpy_data = \
                sample_pole.sample(env_name='cartpole', sample_size=self.sample_size, noise=self.noise)
            data_len = len(x_numpy_data)

            # place holder for data
            data_x = torch.zeros(data_len, 2, self.width, self.width)
            data_u = torch.zeros(data_len, self.action_dim)
            data_x_next = torch.zeros(data_len, 2, self.width, self.width)

            for i in range(data_len):
                data_x[i] = self._process_image(x_numpy_data[i])
                data_u[i] = torch.from_numpy(u_numpy_data[i])
                data_x_next[i] = self._process_image(x_next_numpy_data[i])

            data_set = (data_x, data_u, data_x_next)

            with open(self.data_path + '{:d}_{:.0f}.pt'.format(self.sample_size, self.noise), 'wb') as f:
                torch.save(data_set, f)

class ThreePoleDataset(BaseDataset):
    width = 80
    height = 80 * 2
    action_dim = 3

    def __init__(self, sample_size, noise):
        data_path = 'data/threepole/'
        super(ThreePoleDataset, self).__init__(data_path, sample_size, noise)

    def _process_image(self, img):
        x = torch.zeros(size=(2, self.width, self.width))
        x[0, :, :] = torch.from_numpy(img[:, :, 0])
        x[1, :, :] = torch.from_numpy(img[:, :, 1])
        return x.unsqueeze(0)

    def _process(self):
        if self.check_exists():
            return
        else:
            x_numpy_data, u_numpy_data, x_next_numpy_data, state_numpy_data, state_next_numpy_data = \
                sample_pole.sample(env_name='threepole', sample_size=self.sample_size, noise=self.noise)
            data_len = len(x_numpy_data)

            # place holder for data
            data_x = torch.zeros(data_len, 2, self.width, self.width)
            data_u = torch.zeros(data_len, self.action_dim)
            data_x_next = torch.zeros(data_len, 2, self.width, self.width)

            for i in range(data_len):
                data_x[i] = self._process_image(x_numpy_data[i])
                data_u[i] = torch.from_numpy(u_numpy_data[i])
                data_x_next[i] = self._process_image(x_next_numpy_data[i])

            data_set = (data_x, data_u, data_x_next)

            with open(self.data_path + '{:d}_{:.0f}.pt'.format(self.sample_size, self.noise), 'wb') as f:
                torch.save(data_set, f)