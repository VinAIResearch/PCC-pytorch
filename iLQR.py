import torch
import numpy as np
from PIL import Image
import imageio
import os
import json
import time

from pcc_model import PCC
import data.planar.sample_planar as planar_sampler

np.random.seed(0)
torch.manual_seed(0)

z_dim, u_dim = 2, 2
# define cost matrices
R_z = 10 * torch.eye(z_dim).cuda()
R_u = 1 * torch.eye(u_dim).cuda()
horizon = 40

def draw_start_goal():
    """
    return a random start state (one of three corners), and the goal state (bottom-right corner)
    """
    s_goal = np.random.uniform(planar_sampler.height - planar_sampler.rw - 3,
                                planar_sampler.width - planar_sampler.rw, size=2)
    corner_idx = np.random.randint(0, 3)
    if corner_idx == 0:
        s_start = np.random.uniform(planar_sampler.rw, planar_sampler.rw + 3, size=2)
    if corner_idx == 1:
        x = np.random.uniform(planar_sampler.rw, planar_sampler.rw+3)
        y = np.random.uniform(planar_sampler.width - planar_sampler.rw - 3,
                            planar_sampler.width - planar_sampler.rw)
        s_start = np.array([x, y])
    if corner_idx == 2:
        x = np.random.uniform(planar_sampler.width - planar_sampler.rw - 3,
                            planar_sampler.width - planar_sampler.rw)
        y = np.random.uniform(planar_sampler.rw, planar_sampler.rw+3)
        s_start = np.array([x, y])
    return corner_idx, s_start, s_goal

def random_traj(z_start, horizon, dynamics):
    """
    initialize a random trajectory
    """
    z = z_start
    z_seq = []
    u_seq = []
    for i in range(horizon):
        z_seq.append(z)
        u = torch.empty(1, u_dim).uniform_(-planar_sampler.max_step_len, planar_sampler.max_step_len).cuda()
        u_seq.append(u)
        with torch.no_grad():
            z_next, _, _, _ = dynamics(z, u)
        z = z_next
    z_seq.append(z)
    u_seq.append(None)
    return z_seq, u_seq

def jacobian(dynamics, z, u):
    """
    compute the jacobian of F(z,u) w.r.t z, u
    """
    if dynamics.armotized:
        z_next, _, A, B = dynamics(z, u)
        return A.view(z_dim, z_dim), B.view(u_dim, u_dim)
    z, u = z.squeeze().repeat(z_dim, 1), u.squeeze().repeat(z_dim, 1)
    z = z.detach().requires_grad_(True)
    u = u.detach().requires_grad_(True)
    z_next, _, _, _ = dynamics(z, u)
    grad_inp = torch.eye(z_dim).cuda()
    A, B = torch.autograd.grad(z_next, [z, u], [grad_inp, grad_inp])
    return A, B

def seq_jacobian(dynamics, z_seq, u_seq):
    """
    compute the jacobian w.r.t each pair in the trajectory
    """
    A_seq, B_seq = [], []
    for i in range(len(z_seq) - 1):
        z, u = z_seq[i], u_seq[i]
        A, B = jacobian(dynamics, z, u)
        A_seq.append(A)
        B_seq.append(B)
    return A_seq, B_seq

def compute_loss(R_z, R_u, z_seq, z_goal, u_seq):
    loss = 0.0
    for t in range(len(z_seq) - 1):
        z_t_diff, u_t = z_seq[t] - z_goal, u_seq[t]
        loss += z_t_diff.view(1,-1).mm(R_z).mm(z_t_diff.view(-1,1)) + u_t.view(1,-1).mm(R_u).mm(u_t.view(-1,1))
    z_T_diff = z_seq[-1] - z_goal
    loss += z_T_diff.view(1,-1).mm(R_z).mm(z_T_diff.view(-1,1))
    return loss

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

def backward(R_z, R_u, z_seq, u_seq, z_goal, A_seq, B_seq, V_T_z, V_T_zz):
    """
    do the backward pass
    return a sequence of k and K matrices
    """
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

def forward(z_seq, u_seq, k, K, dynamics):
    """
    update the trajectory, given k and K
    """
    z_seq_new = []
    z_seq_new.append(z_seq[0])
    u_seq_new = []
    for i in range(0, len(z_seq) - 1):
        u_new = u_seq[i] + k[i].view(1,-1) + K[i].mm((z_seq_new[i] - z_seq[i]).view(-1,1)).view(1,-1)
        u_seq_new.append(u_new)
        with torch.no_grad():
            z_new, _, _, _ = dynamics(z_seq_new[i], u_new)
        z_seq_new.append(z_new)
    u_seq_new.append(None)
    return z_seq_new, u_seq_new

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

def iQLR_solver(R_z, R_u, z_seq, z_goal, u_seq, dynamics, iters=10):
    """
    - run backward: linearize around the current trajectory and perform optimal control
    - run forward: update the current trajectory
    - repeat
    """
    old_loss = compute_loss(R_z, R_u, z_seq, z_goal, u_seq)
    V_T_z = 2 * R_z.mm((z_seq[-1] - z_goal).view(-1,1))
    V_T_zz = 2 * R_z
    A_seq, B_seq = seq_jacobian(dynamics, z_seq, u_seq)
    # print ('iLQR loss iter {:02d}: {:05f}'.format(0, old_loss.item()))
    # for i in range(iters):
    #     k, K = backward(R_z, R_u, z_seq, u_seq, z_goal, A_seq, B_seq, V_T_z, V_T_zz)
    #     z_seq, u_seq = forward(z_seq, u_seq, k, K, dynamics)
    #     new_loss = compute_loss(R_z, R_u, z_seq, z_goal, u_seq)
    #     if torch.isnan(new_loss):
    #         return None, None, None, None
    #     print ('iLQR loss iter {:02d}: {:05f}'.format(i+1, new_loss.item()))
    #     old_loss = new_loss
    #     V_T_z = 2 * R_z.mm((z_seq[-1] - z_goal).view(-1,1))
    #     V_T_zz = 2 * R_z
    #     A_seq, B_seq = seq_jacobian(dynamics, z_seq, u_seq)
    # return z_seq, u_seq, k, K
    k, K = backward(R_z, R_u, z_seq, u_seq, z_goal, A_seq, B_seq, V_T_z, V_T_zz)
    return k, K

def reciding_horizon(R_z, R_u, s_start, z_start, z_goal, dynamics, encoder, iters_ilqr, horizon):
    # for the first step
    z_seq, u_seq = random_traj(z_start, horizon, dynamics)
    loss = compute_loss(R_z, R_u, z_seq, z_goal, u_seq)
    print ('Horizon {:02d}: {:05f}'.format(0, loss.item()))
    u_opt = []
    s = s_start
    for i in range(horizon):
        # print ('Horizon {:02d}'.format(i+1))
        # z_seq, u_seq, k, K = iQLR_solver(R_z, R_u, z_seq, z_goal, u_seq, dynamics)
        # if z_seq is None:
        #     return None
        # u_first_opt = u_seq[0] # only apply the first action
        k, K = iQLR_solver(R_z, R_u, z_seq, z_goal, u_seq, dynamics)
        u_first_opt = u_seq[0] + k[0].view(1,-1)
        if torch.any(torch.isnan(u_first_opt)):
            return None
        u_opt.append(u_first_opt)

        # get z_k+1 from the true dynamics
        s = np.round(s + np.array(u_first_opt.squeeze().cpu().detach()))
        next_obs = planar_sampler.render(s)
        next_x = torch.from_numpy(next_obs).cuda().squeeze(0).view(-1, 1600)
        z_start, _ = encoder(next_x)

        # update the nominal trajectory
        z_seq, u_seq = z_seq[1:], u_seq[1:]
        k, K = k[1:], K[1:]
        z_seq, u_seq = update_seq_act(z_seq, z_start, u_seq, k, K, dynamics)
        loss = compute_loss(R_z, R_u, z_seq, z_goal, u_seq)
        if torch.isnan(loss):
            return None
        print ('Horizon {:02d}: {:05f}'.format(i+1, loss.item()))
        # print ('==============================')
    return u_opt

def main():
    device = torch.device("cuda")
    folder = 'result/planar'
    log_folders = [os.path.join(folder, dI) for dI in os.listdir(folder) if os.path.isdir(os.path.join(folder,dI))]
    log_folders.sort()
    avg_model_percent = 0.0
    best_model_percent = 0.0
    for log in log_folders:
        with open(log + '/settings', 'r') as f:
            settings = json.load(f)
            armotized = settings['armotized']

        log_base = os.path.basename(os.path.normpath(log))
        model_path = 'iLQR_gif/' + log_base
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        print ('Performing iLQR for ' + log_base)

        model = PCC(armotized, 1600, 2, 2, 'planar').to(device)
        model.load_state_dict(torch.load(log + '/model_5000'))
        model.eval()
        dynamics = model.dynamics
        encoder = model.encoder

        avg_percent = 0
        for task in range(10):
            print ('Performing task ' + str(task+1))
            # task_path = model_path + '/task_{:01d}'.format(task+1)
            # traj_path = model_path + '/trajectory' + '/task_{:01d}'.format(task+1)
            # if not os.path.exists(traj_path):
            #     os.makedirs(traj_path)
            # draw random initial state and goal state
            corner_idx, s_start, s_goal = draw_start_goal()
            image_start = planar_sampler.render(s_start)
            image_goal = planar_sampler.render(s_goal)
            x_start = torch.from_numpy(image_start).cuda().squeeze(0).view(-1, 1600)
            x_goal = torch.from_numpy(image_goal).cuda().squeeze(0).view(-1, 1600)

            z_start, _ = model.encode(x_start)
            z_goal, _ = model.encode(x_goal)
            u_opt = reciding_horizon(R_z, R_u, s_start, z_start, z_goal, dynamics, encoder, 10, 40)
            if u_opt is None:
                avg_percent += 0
                with open(model_path + '/result.txt', 'a+') as f:
                    f.write('Task {:01d} start at corner {:01d}: '.format(task+1, corner_idx) + ' crashed' + '\n')
                continue
            s = s_start

            images = [Image.fromarray(image_start * 255.).convert('L')]
            # Image.fromarray(image_start * 255.).convert('L').save(traj_path + '/0.png')
            close_steps = 0.0
            for i, u in enumerate(u_opt):
                u = np.array(u.squeeze().cpu().detach())
                s = np.round(s + u)
                if np.sqrt(np.sum(s - s_goal)**2) <= 2:
                    close_steps += 1
                image = planar_sampler.render(s)
                images.append(Image.fromarray(image * 255.).convert('L'))
            percent = close_steps / 40
            avg_percent += percent
            with open(model_path + '/result.txt', 'a+') as f:
                f.write('Task {:01d} start at corner {:01d}: '.format(task+1, corner_idx) + str(close_steps / 40) + '\n')
            images.append(Image.fromarray(image_goal * 255.).convert('L'))
            imageio.mimsave(model_path + '/task_{:01d}.gif'.format(task+1), images)
        
        avg_percent = avg_percent / 10
        avg_model_percent += avg_percent
        if avg_percent > best_model_percent:
            best_model = log_base
            best_model_percent = avg_percent
        with open(model_path + '/result.txt', 'a+') as f:
            f.write('Average percentage: ' + str(avg_percent))

    avg_model_percent = avg_model_percent / len(log_folders)
    with open('iLQR_round_s/result.txt', 'w') as f:
        f.write('Average percentage of all models: ' + str(avg_model_percent) + '\n')
        f.write('Best model: ' + best_model + ', best percentage: ' + str(best_model_percent))
 
if __name__ == '__main__':
    main()