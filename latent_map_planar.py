from PIL import Image, ImageDraw
import numpy as np
from colour import Color
import torch
import argparse
import json
import os
import matplotlib.pyplot as plt

from mdp.plane_obstacles_mdp import PlanarObstaclesMDP
from pcc_model import PCC

blue = Color('blue')
colors = list(blue.range_to(Color("red"), 40))
colors_rgb = [color.rgb for color in colors]
start, end = 0, 40
width, height = 40, 40

# states corresponding to obstacles' positions
def get_invalid_state(mdp):
    invalid_pos = []
    for x in range(start, end):
        for y in range(start, end):
            s = [x,y]
            if not mdp.is_valid_state(np.array(s)):
                invalid_pos.append(s)
    return invalid_pos

def color_gradient():
    img = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(img)
    
    for i, color in zip(range(start, end), colors_rgb):
        r1, g1, b1 = color[0] * 255., color[1] * 255., color[2] * 255.
        draw.line((i, start, i, end), fill=(int(r1), int(g1), int(b1)))

    return img

def get_true_map(mdp):
    invalid_pos = get_invalid_state(mdp)
    color_gradient_img = color_gradient()
    img_scaled = Image.new("RGB", (width * 10, height*10), "#FFFFFF")
    draw = ImageDraw.Draw(img_scaled)
    for y in range(start, end):
        for x in range(start, end):
            if [y, x] in invalid_pos:
                continue
            else:
                x_scaled, y_scaled = x * 10, y * 10
                draw.ellipse((x_scaled-2, y_scaled-2, x_scaled+2, y_scaled+2), fill = color_gradient_img.getpixel((x,y)))
    img_arr_scaled = np.array(img_scaled) / 255.
    return img_arr_scaled

def draw_latent_map(model, mdp):
    invalid_pos = get_invalid_state(mdp)
    img = color_gradient()
    # compute latent z
    all_z = []
    for x in range(start, end):
        for y in range(start, end):
            s = np.array([x,y])
            if [x,y] in invalid_pos:
                all_z.append(np.zeros(2))
            else:
                with torch.no_grad():
                    obs = torch.Tensor(mdp.render(s)).unsqueeze(0).view(-1,1600).double()
                    if next(model.parameters()).is_cuda:
                        obs = obs.cuda()
                    mu = model.encode(obs).mean
                z = mu.squeeze().cpu().numpy()
                all_z.append(np.copy(z))
    all_z = np.array(all_z)

    # normalize and scale to plot
    z_min = np.min(all_z, axis = 0)
    all_z = np.round(20 * (all_z - z_min) + 30).astype(np.int)

    # plot
    latent_map = {}
    i = 0
    for x in range(start, end):
        for y in range(start, end):
            latent_map[(x,y)] = all_z[i]
            i += 1
            
    img_latent = Image.new("RGB", (mdp.width * 10, mdp.height * 10), "#FFFFFF")
    draw = ImageDraw.Draw(img_latent)
    for k in latent_map:
        x, y = k
        if [x, y] in invalid_pos:
            continue
        else:
            x_scaled, y_scaled = latent_map[k][1], latent_map[k][0]
            draw.ellipse((x_scaled-2, y_scaled-2, x_scaled+2, y_scaled+2), fill = img.getpixel((y, x)))
    return img_latent

def show_latent_map(model, mdp):
    true_map = get_true_map(mdp)
    latent_map = draw_latent_map(model, mdp)
    latent_map = np.array(latent_map) / 255.0

    f, axarr = plt.subplots(1, 2, figsize=(15, 15))
    axarr[0].imshow(true_map)
    axarr[1].imshow(latent_map)
    plt.show()

def main(args):
    log_path = args.log_path
    epoch = args.epoch

    mdp = PlanarObstaclesMDP()

    # load the specified model
    with open(log_path + '/settings', 'r') as f:
        settings = json.load(f)
    armotized = settings['armotized']
    model = PCC(armotized, 1600, 2, 2, 'planar')
    model.load_state_dict(torch.load(log_path + '/model_' + str(epoch), map_location='cpu'))
    model.eval()

    show_latent_map(model, mdp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train pcc model')

    parser.add_argument('--log_path', required=True, type=str, help='path to trained model')
    parser.add_argument('--epoch', required=True, type=int, help='load model corresponding to this epoch')
    args = parser.parse_args()

    main(args)

# from mdp.plane_obstacles_mdp import PlanarObstaclesMDP
# from pcc_model import PCC
# mdp = PlanarObstaclesMDP()
# start = 0
# end = 39
# invalid_pos = get_invalid_state(mdp, start, end)
# img_arr, img = random_gradient(start, end, mdp.width, mdp.height, invalid_pos)
# get_true_map(mdp, start, end, mdp.width, mdp.height, img)

# mdp = PlanarObstaclesMDP()
# model = PCC(armotized=False, x_dim=1600, z_dim=2, u_dim=2, env = 'planar').cuda()
# model.load_state_dict(torch.load('./new_mdp_result/planar/log_10/model_5000'))
# latent_map = draw_latent_map(model, mdp)
# latent_map.show()
