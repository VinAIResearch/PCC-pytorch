import torch
import numpy as np
from PIL import Image
import imageio
import os
import json
import argparse
from torchvision.transforms import ToTensor

from pcc_model import PCC
import data.sample_planar as planar_sampler
from mdp.plane_obstacles_mdp import PlanarObstaclesMDP
from gym.envs.classic_control import PendulumEnv
import data.sample_pendulum as pendulum_sampler
from ilqr_utils import *
from datasets import *

np.random.seed(0)
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

samplers = {'planar': planar_sampler, 'pendulum': pendulum_sampler}
mdps = {'planar': PlanarObstaclesMDP, 'pendulum': PendulumEnv}
network_dims = {'planar': (1600, 2, 2), 'pendulum': (4608, 3, 1)}
img_dims = {'planar': (40, 40), 'pendulum': (48, 96)}
horizons = {'planar': 40, 'pendulum': 400}
K_z = 10
K_u = 1

def one_step_back(R_z, R_u, z, u, z_goal, A, B, V_next_z, V_next_zz):
    """
    V_next_z: first order derivative of the value function at time step t+1
    V_next_zz: second order derivative of the value function at time tep t+1
    A: derivative of F(z, u) w.r.t z at z_bar_t, u_bar_t
    B: derivative of F(z, u) w.r.t u at z_bar_t, u_bar_t
    """
    # compute Q_z, Q_u, Q_zz, Q_uu, Q_uz using cost function, A, B and V
    Q_z = 2 * R_z.mm((z - z_goal).view(-1,1)) + A.transpose(1,0).mm(V_next_z.view(-1,1))
    Q_u = 2 * R_u.mm(u.view(-1,1)) + B.transpose(1,0).mm(V_next_z.view(-1,1))
    Q_zz = 2 * R_z + A.transpose(1,0).mm(V_next_zz).mm(A)
    Q_uz = 0 + B.transpose(1,0).mm(V_next_zz).mm(A)
    Q_uu = 2 * R_u + B.transpose(1,0).mm(V_next_zz).mm(B)
    # compute k and K matrix
    Q_uu_in = torch.inverse(Q_uu)
    k = -Q_uu_in.mm(Q_u)
    K = -Q_uu_in.mm(Q_uz)
    # compute V_z and V_zz using k and K
    V_z = Q_z.transpose(1,0) - k.transpose(1,0).mm(Q_uu).mm(K)
    V_zz = Q_zz - K.transpose(1,0).mm(Q_uu).mm(K)
    return k, K, V_z, V_zz

def iQLR_solver(R_z, R_u, z_seq, z_goal, u_seq, dynamics):
    """
    - run backward: linearize around the current trajectory and perform optimal control
    - run forward: update the current trajectory
    - repeat
    """
    old_loss = compute_loss(R_z, R_u, z_seq, z_goal, u_seq)
    V_T_z = 2 * R_z.mm((z_seq[-1] - z_goal).view(-1,1))
    V_T_zz = 2 * R_z
    A_seq, B_seq = seq_jacobian(dynamics, z_seq, u_seq)
    V_next_z, V_next_zz = V_T_z, V_T_zz
    k, K = [], []
    act_seq_len = len(z_seq)
    for i in range(2, act_seq_len + 1):
        t = act_seq_len - i
        k_t, K_t, V_t_z, V_t_zz = one_step_back(R_z, R_u, z_seq[t], u_seq[t], z_goal, A_seq[t], B_seq[t], V_next_z, V_next_zz)
        k.insert(0, k_t)
        K.insert(0, K_t)
        V_next_z, V_next_zz = V_t_z, V_t_zz
    return k, K

def update_seq_act(z_seq, z_start, u_seq, k, K, dynamics):
    """
    update the trajectory, given k and K
    """
    z_seq_new = []
    z_seq_new.append(z_start)
    u_seq_new = []
    for i in range(0, len(z_seq) - 1):
        u_new = u_seq[i] + k[i].view(1,-1) + K[i].mm((z_seq_new[i] - z_seq[i]).view(-1,1)).view(1,-1)
        u_seq_new.append(u_new)
        with torch.no_grad():
            z_new, _, _, _ = dynamics(z_seq_new[i], u_new)
        z_seq_new.append(z_new)
    u_seq_new.append(None)
    return z_seq_new, u_seq_new

def compute_loss(R_z, R_u, z_seq, z_goal, u_seq):
    loss = 0.0
    for t in range(len(z_seq) - 1):
        z_t_diff, u_t = z_seq[t] - z_goal, u_seq[t]
        loss += z_t_diff.view(1,-1).mm(R_z).mm(z_t_diff.view(-1,1)) + u_t.view(1,-1).mm(R_u).mm(u_t.view(-1,1))
    z_T_diff = z_seq[-1] - z_goal
    loss += z_T_diff.view(1,-1).mm(R_z).mm(z_T_diff.view(-1,1))
    return loss

def reciding_horizon(env_name, mdp, sampler, img_dim, x_dim, R_z, R_u, s_start, z_start, z_goal, dynamics, encoder, horizon):
    # for the first step
    z_dim, u_dim = R_z.size(0), R_u.size(0)
    z_seq, u_seq = random_traj(mdp, s_start, z_start, horizon, dynamics)
    loss = compute_loss(R_z, R_u, z_seq, z_goal, u_seq)
    print ('Horizon {:02d}: {:05f}'.format(0, loss.item()))
    u_opt = []
    s = s_start
    for i in range(horizon):
        # optimal perturbed policy at time step k
        k, K = iQLR_solver(R_z, R_u, z_seq, z_goal, u_seq, dynamics)
        u_first_opt = u_seq[0] + k[0].view(1,-1)
        if torch.any(torch.isnan(u_first_opt)):
            return None
        u_opt.append(u_first_opt)

        # get z_k+1 from the true dynamics
        if env_name == 'planar':
            s = mdp.transition_function(s, u_first_opt.squeeze().cpu().detach())
            # s = s + np.array(u_first_opt.squeeze().cpu().detach())
            next_obs = mdp.render(s)
            next_x = torch.from_numpy(next_obs).cuda().squeeze(0).view(-1, x_dim)
        elif env_name == 'pendulum':
            s = mdp.step_from_state(s, u_first_opt.squeeze().cpu().detach().numpy())
            next_obs_1, next_obs_2 = sampler.render(mdp, s)
            next_obs = Image.fromarray(np.hstack((next_obs_1, next_obs_2)))
            next_x = ToTensor()((Image.fromarray(next_obs).convert('L').
                        resize((img_dim[0], img_dim[1])))).cuda().transpose(-1,-2).double()
            print (next_x.size())
        # next_obs = env.render_state(s)
        # next_x = torch.from_numpy(next_obs).cuda().squeeze(0).view(-1, 1600)
        z_start, _ = encoder(next_x)

        # update the nominal trajectory
        z_seq, u_seq = z_seq[1:], u_seq[1:]
        k, K = k[1:], K[1:]
        z_seq, u_seq = update_seq_act(z_seq, z_start, u_seq, k, K, dynamics)
        loss = compute_loss(R_z, R_u, z_seq, z_goal, u_seq)
        if torch.isnan(loss):
            return None
        print ('Horizon {:02d}: {:05f}'.format(i+1, loss.item()))
    return u_opt

def main(args):
    env_name = args.env

    sampler = samplers[env_name]
    mdp = mdps[env_name]()
    x_dim, z_dim, u_dim = network_dims[env_name]
    img_dim = img_dims[env_name]
    horizon = horizons[env_name]
    R_z = K_z * torch.eye(z_dim).cuda()
    R_u = K_u * torch.eye(u_dim).cuda()

    folder = 'new_mdp_result/' + env_name
    log_folders = [os.path.join(folder, dI) for dI in os.listdir(folder) if os.path.isdir(os.path.join(folder,dI))]
    log_folders.sort()
    # print (log_folders)

    device = torch.device("cuda")
    
    avg_model_percent = 0.0
    best_model_percent = 0.0
    for log in log_folders:
        with open(log + '/settings', 'r') as f:
            settings = json.load(f)
            armotized = settings['armotized']

        log_base = os.path.basename(os.path.normpath(log))
        model_path = 'iLQR_new_mdp_result/' +  env_name + '/' + log_base
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        print ('Performing iLQR for ' + log_base)

        model = PCC(armotized, x_dim, z_dim, u_dim, env_name).to(device)
        model.load_state_dict(torch.load(log + '/model_5000'))
        model.eval()
        dynamics = model.dynamics
        encoder = model.encoder

        avg_percent = 0
        for task in range(10):
            print ('Performing task ' + str(task+1))
            # draw random initial state and goal state
            corner_idx, s_start, s_goal = random_start_goal(env_name, mdp)
            if env_name == 'planar':
                image_start = mdp.render(s_start)
                image_goal = mdp.render(s_goal)
                x_start = torch.from_numpy(image_start).cuda().squeeze(0).view(-1, x_dim)
                x_goal = torch.from_numpy(image_goal).cuda().squeeze(0).view(-1, x_dim)
            elif env_name == 'pendulum':
                image_start_1, image_start_2 = sampler.render(mdp, s_start)
                image_start = Image.fromarray(np.hstack((image_start_1, image_start_2)))
                image_goal_1, image_goal_2 = sampler.render(mdp, s_goal)
                image_goal = Image.fromarray(np.hstack((image_goal_1, image_goal_2)))

                # print ('x dim ' + str(x_dim))
                x_start = ToTensor()(image_start.convert('L').
                            resize((img_dim[0], img_dim[1]))).cuda().transpose(-1,-2).reshape(-1, x_dim).double()
                # print ('x start ' + str(x_start.size()))
                x_goal = ToTensor()(image_goal.convert('L').
                            resize((img_dim[0], img_dim[1]))).cuda().transpose(-1,-2).reshape(-1, x_dim).double()
                # print ('x goal ' + str(x_goal.size()))
            z_start, _ = model.encode(x_start)
            z_goal, _ = model.encode(x_goal)

            # perform optimal control for this task
            u_opt = reciding_horizon(env_name, mdp, sampler, img_dim, x_dim, R_z, R_u, s_start, z_start, z_goal, dynamics, encoder, horizon)
            if u_opt is None:
                avg_percent += 0
                with open(model_path + '/result.txt', 'a+') as f:
                    f.write('Task {:01d} start at corner {:01d}: '.format(task+1, corner_idx) + ' crashed' + '\n')
                continue

            # compute the trajectory
            s = s_start
            images = [image_start]
            close_steps = 0.0
            for i, u in enumerate(u_opt):
                u = np.array(u.squeeze().cpu().detach())
                if env_name == 'planar':
                    s = mdp.transition_function(s, u)
                    image = mdp.render(s)
                    images.append(image)
                elif env_name == 'pendulum':
                    s = mdp.step_from_state(s, u)
                    image = mdp.render_state(s[0])
                    images.append(image)
                if is_close_goal(env_name, s, s_goal):
                    close_steps += 1

            # compute the percentage close to goal
            percent = close_steps / 40
            avg_percent += percent
            with open(model_path + '/result.txt', 'a+') as f:
                f.write('Task {:01d} start at corner {:01d}: '.format(task+1, corner_idx) + str(close_steps / 40) + '\n')

            # save trajectory as gif file
            gif_path = model_path + '/task_{:01d}.gif'.format(task+1)
            save_traj(images, image_goal, gif_path)
        
        avg_percent = avg_percent / 10
        avg_model_percent += avg_percent
        if avg_percent > best_model_percent:
            best_model = log_base
            best_model_percent = avg_percent
        with open(model_path + '/result.txt', 'a+') as f:
            f.write('Average percentage: ' + str(avg_percent))

    avg_model_percent = avg_model_percent / len(log_folders)
    with open('iLQR_new_mdp_result/result.txt', 'w') as f:
        f.write('Average percentage of all models: ' + str(avg_model_percent) + '\n')
        f.write('Best model: ' + best_model + ', best percentage: ' + str(best_model_percent))
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train pcc model')
    parser.add_argument('--env', required=True, type=str, help='environment used for training')

    args = parser.parse_args()

    main(args)
