import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from matplotlib.animation import FuncAnimation, writers
import colour
from random import randint as rint
import numpy as np
from colour import Color
import torch
import os

#from e2c_model import E2C
#from data.sample_planar import *

red = Color('blue')
colors = list(red.range_to(Color("red"),36))
colors_rgb = [color.rgb for color in colors]

def get_invalid_state(mdp, start, end, width, height):
    invalid_pos = []
    for x in range(start, end + 1):
        for y in range(start, end + 1):
            s = [x,y]
            if not mdp.is_valid_state(np.array(s)):
                invalid_pos.append(s)
    return invalid_pos

def random_gradient(start, end, width, height, invalid_pos):
    img = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(img)
    
    for i, color in zip(range(start, end+1), colors_rgb):
        r1, g1, b1 = color[0] * 255., color[1] * 255., color[2] * 255.
        draw.line((i,start,i,end), fill=(int(r1), int(g1), int(b1)))

    img_arr = np.array(img)
    for x, y in invalid_pos:
        img_arr[x, y] = 255.
    return img_arr / 255., img

def get_true_map(mdp, start, end, width, height, img):
    img_scaled = Image.new("RGB", (width * 10, height*10), "#FFFFFF")
    draw = ImageDraw.Draw(img_scaled)
    invalid_pos = get_invalid_state(mdp, start, end, width, height)
    for y in range(start, end + 1):
        for x in range(start, end + 1):
            if [y, x] in invalid_pos:
                continue
            else:
                x_scaled, y_scaled = x * 10, y * 10
                draw.ellipse((x_scaled-2, y_scaled-2, x_scaled+2, y_scaled+2), fill = img.getpixel((x,y)))
    img_scaled.save('map.png', 'PNG')
    img_arr_scaled = np.array(img_scaled) / 255.
    return img_arr_scaled

def draw_latent_map(model, mdp):
    model.eval()

    start = int(np.round(mdp.half_agent_size))
    end = int(np.round(mdp.width - mdp.half_agent_size))
    width, height = mdp.width, mdp.height

    invalid_pos = get_invalid_state(mdp, start, end, width, height)
    img_arr, img = random_gradient(start, end, width, height, invalid_pos)
    # compute latent z
    all_z = []
    for x in range(start, end + 1):
        for y in range(start, end + 1):
            s = np.array([x,y])
            if [x,y] in invalid_pos:
                all_z.append(np.zeros(2))
            else:
                obs = mdp.render(s)
                with torch.no_grad():
                    obs = torch.Tensor(mdp.render(s)).unsqueeze(0).view(-1,1600).double().cuda(0)
                    mu, sigma = model.encode(obs)
                z = mu.squeeze().cpu().numpy()
                all_z.append(np.copy(z))
    all_z = np.array(all_z)

    # normalize and scale to plot
    z_min = np.min(all_z, axis = 0)
    all_z = np.round(15 * (all_z - z_min) + 30).astype(np.int)

    # plot
    latent_map = {}
    i = 0
    for x in range(start, end + 1):
        for y in range(start, end + 1):
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

# from mdp.plane_obstacles_mdp import PlanarObstaclesMDP

# mdp = PlanarObstaclesMDP()
# start = int(np.round(mdp.half_agent_size))
# end = int(np.round(mdp.width - mdp.half_agent_size))
# width, height = mdp.width, mdp.height
# invalid_pos = get_invalid_state(mdp, start, end, width, height)
# img_arr, img = random_gradient(start, end, width, height, invalid_pos)
# get_true_map(mdp, start, end, width, height, img)