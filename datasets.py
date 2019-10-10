import os
from os import path
from PIL import Image
import numpy as np
import json
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from tqdm import tqdm
import pickle
import torch

torch.set_default_dtype(torch.float64)

class PlanarDataset(Dataset):
    width = 40
    height = 40
    action_dim = 2
    ratio = 5/6 # raito of training set
    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(self, root, train = True):
        self.root = root
        self.train = train
        self.raw_folder = path.join(self.root, 'raw')
        self.processed_folder = path.join(self.root, 'processed')
        if not path.exists(self.processed_folder):
            os.makedirs(self.processed_folder)
        self._process()
        
        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.data_x, self.data_u, self.data_x_next = torch.load(path.join(self.processed_folder, data_file))

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, index):
        return self.data_x[index], self.data_u[index], self.data_x_next[index]

    def check_exists(self):
        return (path.exists(path.join(self.processed_folder,
                                            self.training_file)) and
                path.exists(path.join(self.processed_folder,
                                            self.test_file)))

    def _process_image(self, img):
        return ToTensor()(img.convert('L').
                           resize((self.width,
                                   self.height)))

    def _process(self):
        if self.check_exists():
            return
        else:
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

            train_len = int(self.ratio * data_len)
            training_set = (data_x[:train_len], data_u[:train_len], data_x_next[:train_len])
            test_set = (data_x[train_len:], data_u[train_len:], data_x_next[train_len:])

            with open(path.join(self.processed_folder, self.training_file), 'wb') as f:
                torch.save(training_set, f)
            with open(path.join(self.processed_folder, self.test_file), 'wb') as f:
                torch.save(test_set, f)

class GymPendulumDatasetV2(Dataset):
    width = 48 * 2
    height = 48
    action_dim = 1

    def __init__(self, dir):
        self.dir = dir
        with open(path.join(dir, 'data.json')) as f:
            self._data = json.load(f)
        self._process()

    def __len__(self):
        return len(self._data['samples'])

    def __getitem__(self, index):
        return self._processed[index]

    @staticmethod
    def _process_image(img):
        return ToTensor()((img.convert('L').
                           resize((GymPendulumDatasetV2.width,
                                   GymPendulumDatasetV2.height))))

    def _process(self):
        preprocessed_file = os.path.join(self.dir, 'processed.pkl')
        if not os.path.exists(preprocessed_file):
            processed = []
            for sample in tqdm(self._data['samples'], desc='processing data'):
                before = Image.open(os.path.join(self.dir, sample['before']))
                after = Image.open(os.path.join(self.dir, sample['after']))

                processed.append((self._process_image(before),
                                  np.array(sample['control']),
                                  self._process_image(after)))

            with open(preprocessed_file, 'wb') as f:
                pickle.dump(processed, f)
            self._processed = processed
        else:
            with open(preprocessed_file, 'rb') as f:
                self._processed = pickle.load(f)

# planar = PlanarDataset('data/planar')
# print (planar[0][0].device)