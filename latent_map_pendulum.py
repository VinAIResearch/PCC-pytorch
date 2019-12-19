import numpy as np
from colour import Color
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torchvision.transforms import ToTensor
import json
import argparse
import plotly.express as px

from mdp.pendulum_mdp import PendulumMDP
from mdp.pendulum_gym import PendulumGymMDP
from pcc_model import PCC

red = Color('red')
blue = Color('blue')
num_angles = 100
num_each_angle = 20

np.random.seed(0)
torch.manual_seed(0)

def map_angle_color(num_angles, mdp):
    colors = list(red.range_to(blue, num_angles))
    colors_rgb = [color.rgb for color in colors]
    all_angles = np.linspace(start=mdp.angle_range[0], stop=mdp.angle_range[1], num=num_angles)
    angle_color_map = dict(zip(all_angles, colors_rgb))
    return angle_color_map, colors_rgb

def assign_latent_color(model, angel, mdp):
    # the same angle corresponds to multiple states -> multiple latent vectors
    # map an angle to multiple latent vectors corresponding to that angle
    angle_vels = np.linspace(start=mdp.angular_velocity_range[0],
                             stop=mdp.angular_velocity_range[1], num=num_each_angle)
    all_z_for_angle = []
    for i in range(num_each_angle):
        ang_velocity = angle_vels[i]
        s = np.array([angel, ang_velocity])
        x = mdp.render(s).squeeze()
        # take a random action
        u = mdp.sample_random_action()
        s_next = mdp.transition_function(s, u)
        x_next = mdp.render(s_next).squeeze()
        # reverse order: the state we want to represent is x not x_next
        x_with_history = np.vstack((x_next, x))
        x_with_history = ToTensor()(x_with_history).double()
        with torch.no_grad():
            z = model.encode(x_with_history.view(-1, x_with_history.shape[-1] * x_with_history.shape[-2])).mean
        all_z_for_angle.append(z.detach().squeeze().numpy())
    return all_z_for_angle

def show_latent_map(model, mdp):
    angle_color_map, colors_rgb = map_angle_color(num_angles, mdp)
    colors_list = []
    for color in colors_rgb:
        for i in range(num_each_angle):
            colors_list.append(list(color))
    all_z = []

    for angle in angle_color_map:
        all_z_for_angle = assign_latent_color(model, angle, mdp)
        all_z += all_z_for_angle
    all_z = np.array(all_z)

    z_min = np.min(all_z, axis=0)
    z_max = np.max(all_z, axis=0)
    all_z = 2 * (all_z - z_min) / (z_max - z_min) - 1.0
    all_z = all_z * 35

    ax = plt.axes(projection='3d')
    ax.set_xlim([-100, 100])
    ax.set_ylim([-100, 100])
    ax.set_zlim([-100, 100])
    xdata = all_z[:, 0]
    ydata = all_z[:, 1]
    zdata = all_z[:, 2]

    # colors_list = np.array(colors_list)
    # all_z = np.concatenate((all_z, colors_list), axis=1)
    # print (all_z.shape)

    ax.scatter(xdata, ydata, zdata, c=colors_list, marker='o', s=10)
    plt.show()
    # fig = px.scatter_3d(all_z, x=0, y=1, z=2,
    #                     color=[3,4,5])
    # fig.show()

def main(args):
    gym = args.gym
    log_path = args.log_path
    epoch = args.epoch

    if not gym:
        mdp = PendulumMDP()
    else:
        mdp = PendulumGymMDP()
    # load the specified model
    with open(log_path + '/settings', 'r') as f:
        settings = json.load(f)
    armotized = settings['armotized']
    model = PCC(armotized=armotized, x_dim=4608, z_dim=3, u_dim=1, env='pendulum')
    model.load_state_dict(torch.load(log_path + '/model_' + str(epoch), map_location='cpu'))
    model.eval()

    show_latent_map(model, mdp)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train pcc model')

    parser.add_argument('--gym', required=True, type=str2bool, nargs='?',
                        const=True, default=False, help='Pendulum Gym or not')
    parser.add_argument('--log_path', required=True, type=str, help='path to trained model')
    parser.add_argument('--epoch', required=True, type=int, help='load model corresponding to this epoch')
    args = parser.parse_args()

    main(args)