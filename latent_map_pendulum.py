import numpy as np
from colour import Color
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor

from mdp.pendulum_mdp import PendulumMDP
from pcc_model import PCC

red = Color('red')
blue = Color('blue')
mdp = PendulumMDP()

np.random.seed(0)
torch.manual_seed(0)

def draw_true_map(num_angles):
    colors = list(red.range_to(blue, num_angles))
    colors_rgb = [color.rgb for color in colors]
    all_angles = np.linspace(start=mdp.angle_range[0], stop=mdp.angle_range[1], num=num_angles)
    angle_color_map = dict(zip(all_angles, colors_rgb))
    return angle_color_map, colors_rgb

def assign_latent_color(pcc_model, angel, num_state_each_angle):
    # the same angle corresponds to multiple states -> multiple latent vectors
    # map an angle to multiple latent vectors corresponding to that angle
    angle_vels = np.linspace(start=mdp.angular_velocity_range[0],
                             stop=mdp.angular_velocity_range[1], num=num_state_each_angle)
    all_z_for_angle = []
    for i in range(num_state_each_angle):
        ang_velocity = angle_vels[i]
        s = np.array([angel, ang_velocity])
        x = mdp.render(s).squeeze()
        # take a random action
        u = mdp.sample_random_action()
        # u = np.array([0.0])
        s_next = mdp.transition_function(s, u)
        x_next = mdp.render(s_next).squeeze()
        # reverse order: the state we want to represent is x not x_next
        x_with_history = np.vstack((x_next, x))
        x_with_history = ToTensor()(x_with_history).double()
        with torch.no_grad():
            z, _ = pcc_model.encode(x_with_history.view(-1, x_with_history.shape[-1] * x_with_history.shape[-2]))
        all_z_for_angle.append(z.detach().squeeze().numpy())
    return all_z_for_angle

model = PCC(armotized=False, x_dim=4608, z_dim=3, u_dim=1, env='pendulum')
model.load_state_dict(torch.load('result/pendulum/new_dataset_10/model_5000', map_location='cpu'))
model.eval()
num_angles = 50
num_obs_each_angle = 20
angle_color_map, colors_rgb = draw_true_map(num_angles)
colors_list = []
for color in colors_rgb:
    for i in range(num_obs_each_angle):
        colors_list.append(list(color))
all_z = []
counter = 0
for angle in angle_color_map:
    all_z_for_angle = assign_latent_color(model, angle, num_obs_each_angle)
    all_z += all_z_for_angle
    # break
    # counter += 1
    # if counter == 2:
    #     break
# all_z = assign_latent_color(model, list(angle_color_map.keys())[3], 50)
all_z = np.array(all_z)
# print (all_z)
z_min = np.min(all_z, axis=0)
z_max = np.max(all_z, axis=0)
all_z = 2 * (all_z - z_min) / (z_max - z_min) - 1.0
z_min = np.min(all_z, axis=0)
z_max = np.max(all_z, axis=0)
print (z_min)
print (z_max)
all_z = all_z * 35
# all_z = np.round(all_z * 100).astype(int)


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlim([-100, 100])
ax.set_ylim([-100, 100])
ax.set_zlim([-100, 100])
xdata = all_z[:, 0]
ydata = all_z[:, 1]
zdata = all_z[:, 2]
colors = ['red'] * 10 + ['blue'] * 10

ax.scatter(xdata, ydata, zdata, c=colors_list, marker='o', s=10)
plt.show()