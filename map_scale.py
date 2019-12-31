import torch
from torch.utils.data import DataLoader
import random
import argparse
import json

from pcc_model import PCC
from datasets import *

random.seed(0)
torch.manual_seed(0)

dataset = CartPoleDataset(sample_size=15000, noise=0)
data_loader = DataLoader(dataset, batch_size=128, shuffle=False, drop_last=False, num_workers=4)

def compute_avg_norm_2(model):
    num_batches = len(data_loader)
    avg_norm_2 = 0.0

    for x, _, __ in data_loader:
        z = model.encode(x).mean
        avg_norm_2 += torch.mean(torch.sum(torch.pow(z,2), dim=1))

    avg_norm_2 /= num_batches
    print ('avg norm 2: ' + str(avg_norm_2.item()))

def main(args):
    log_path = args.log_path
    epoch = args.epoch

    # load the specified model
    with open(log_path + '/settings', 'r') as f:
        settings = json.load(f)
    armotized = settings['armotized']
    model = PCC(armotized=armotized, x_dim=(2, 80, 80), z_dim=8, u_dim=1, env='cartpole')
    model.load_state_dict(torch.load(log_path + '/model_' + str(epoch), map_location='cpu'))
    model.eval()

    compute_avg_norm_2(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train pcc model')

    parser.add_argument('--log_path', required=True, type=str, help='path to trained model')
    parser.add_argument('--epoch', required=True, type=int, help='load model corresponding to this epoch')
    args = parser.parse_args()

    main(args)
