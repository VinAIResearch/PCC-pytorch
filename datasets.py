import os
from os import path
from PIL import Image
import numpy as np
import json
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from tqdm import tqdm
import torch

torch.set_default_dtype(torch.float64)

class PlanarDataset(Dataset):
    width = 40
    height = 40
    action_dim = 2
    data_file = 'data.pt'

    def __init__(self, root_path):
        self.root_path = root_path
        self.raw_folder = path.join(self.root_path, 'raw')
        self._process()
        
        self.data_x, self.data_u, self.data_x_next = torch.load(path.join(self.root_path, self.data_file))

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, index):
        return self.data_x[index], self.data_u[index], self.data_x_next[index]

    def _process_image(self, img):
        return ToTensor()(img.convert('L').
                           resize((self.width,
                                   self.height)))

    def check_exists(self):
        return (path.exists(path.join(self.root_path, self.data_file)))

    def _process(self):
        if self.check_exists():
            return
        with open(path.join(self.raw_folder, 'data.json')) as f:
            self.data_json = json.load(f)
        data_len = len(self.data_json['samples'])

        data_x = torch.zeros(data_len, self.width, self.height)
        data_u = torch.zeros(data_len, self.action_dim)
        data_x_next = torch.zeros(data_len, self.width, self.height)

        i = 0
        for sample in tqdm(self.data_json['samples'], desc='processing data'):
            before = Image.open(path.join(self.raw_folder, sample['before']))
            after = Image.open(path.join(self.raw_folder, sample['after']))

            data_x[i] = self._process_image(before)
            data_u[i] = torch.from_numpy(np.array(sample['control']))
            data_x_next[i] = self._process_image(after)
            i += 1

        data_set = (data_x, data_u, data_x_next)

        with open(path.join(self.root_path, self.data_file), 'wb') as f:
            torch.save(data_set, f)

class PendulumDataset(Dataset):
    width = 48
    height = 48 * 2
    action_dim = 1
    data_file = 'data.pt'

    def __init__(self, root_path):
        self.root_path = root_path
        self.raw_folder = path.join(self.root_path, 'raw')
        self._process()

        self.data_x, self.data_u, self.data_x_next = torch.load(path.join(self.root_path, self.data_file))

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, index):
        return self.data_x[index], self.data_u[index], self.data_x_next[index]

    def check_exists(self):
        return (path.exists(path.join(self.root_path, self.data_file)))

    def _process_image(self, img):
        img_tensor =  ToTensor()(img.convert('L').
                          resize((self.height,
                                  self.width)))
        img_tensor = torch.cat((img_tensor[:, :, :self.width], img_tensor[:, :, self.width:]), dim = 1)
        return img_tensor

    def _process(self):
        if self.check_exists():
            return
        else:
            with open(path.join(self.raw_folder, 'data.json')) as f:
                self.data_json = json.load(f)
            data_len = len(self.data_json['samples'])

            data_x = torch.zeros(data_len, self.height, self.width)
            data_u = torch.zeros(data_len, self.action_dim)
            data_x_next = torch.zeros(data_len, self.height, self.width)

            i = 0
            for sample in tqdm(self.data_json['samples'], desc='processing data'):
                before = Image.open(path.join(self.raw_folder, sample['before']))
                after = Image.open(path.join(self.raw_folder, sample['after']))

                data_x[i] = self._process_image(before)
                data_u[i] = torch.from_numpy(np.array(sample['control']))
                data_x_next[i] = self._process_image(after)
                i += 1

            data_set = (data_x, data_u, data_x_next)
            with open(path.join(self.root_path, self.data_file), 'wb') as f:
                torch.save(data_set, f)

# pendulum = PendulumDataset('data/pendulum')
# print (pendulum[0][0])