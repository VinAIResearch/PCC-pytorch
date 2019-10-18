import torch
import numpy as np
from PIL import Image
import os
import json
import time

from pcc_model import PCC
import data.planar.sample_planar as planar_sampler

z_dim, u_dim = 2, 2
# define cost matrices
R_z = 0.1 * torch.eye(z_dim).cuda()
R_u = torch.eye(u_dim).cuda()
# R_o = torch.eye(z_dim) # penalize proximity to obstacles
horizon, act_seq_len = 40, 40

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
    dim_outputs = z.size(1)
    if dynamics.armotized:
        z_next, _, A, B = dynamics(z, u)
        return A.view(z_dim, z_dim), B.view(u_dim, u_dim)
    z, u = z.squeeze().repeat(dim_outputs, 1), u.squeeze().repeat(dim_outputs, 1)
    z = z.detach().requires_grad_(True)
    u = u.detach().requires_grad_(True)
    z_next, _, _, _ = dynamics(z, u)
    grad_inp = torch.eye(dim_outputs).cuda()
    A = torch.autograd.grad(z_next, z, grad_inp, retain_graph=True)[0]
    B = torch.autograd.grad(z_next, u, grad_inp, retain_graph=True)[0]
    return A, B
    # return torch.randn(2,2).cuda(), torch.randn(2,2).cuda()

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
        # print ('Backward step ' + str(i-2))
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
        # print ('Forward step ' + str(i))
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
    print ('iLQR loss iter {:02d}: {:05f}'.format(0, old_loss.item()))
    for i in range(iters):
        k, K = backward(R_z, R_u, z_seq, u_seq, z_goal, A_seq, B_seq, V_T_z, V_T_zz)
        z_seq, u_seq = forward(z_seq, u_seq, k, K, dynamics)
        new_loss = compute_loss(R_z, R_u, z_seq, z_goal, u_seq)
        print ('iLQR loss iter {:02d}: {:05f}'.format(i+1, new_loss.item()))
        old_loss = new_loss
        V_T_z = 2 * R_z.mm((z_seq[-1] - z_goal).view(-1,1))
        V_T_zz = 2 * R_z
        print ('iLQR step ' + str(i))
        A_seq, B_seq = seq_jacobian(dynamics, z_seq, u_seq)
    return z_seq, u_seq, k, K

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

def reciding_horizon(R_z, R_u, s_start, z_start, z_goal, dynamics, encoder, iters_ilqr, horizon):
    # for the first step
    z_seq, u_seq = random_traj(z_start, horizon, dynamics)
    u_opt = []
    s = s_start
    for i in range(horizon):
        print ('Horizon {:02d}'.format(i+1))
        z_seq, u_seq, k, K = iQLR_solver(R_z, R_u, z_seq, z_goal, u_seq, dynamics)
        u_first_opt = u_seq[0] # only apply the first action
        u_opt.append(u_first_opt)
        # get z_k+1 from the true dynamics
        s = s + np.array(u_first_opt.squeeze().cpu().detach())
        image_start = planar_sampler.render(s)
        x_start = torch.from_numpy(image_start).cuda().squeeze(0).view(-1, 1600)
        z_start, _ = encoder(x_start)
        # update the nominal trajectory
        z_seq, u_seq = z_seq[1:], u_seq[1:]
        z_seq, u_seq = update_seq_act(z_seq, z_start, u_seq, k, K, dynamics)
        print ('==============================')
    return u_opt

def main():
    device = torch.device("cuda")
    env_path = os.path.dirname(os.path.abspath(__file__))
    folder = 'result/planar'
    log_folders = [os.path.join(folder, dI) for dI in os.listdir(folder) if os.path.isdir(os.path.join(folder,dI))]
    log_folders.sort()
    for log in log_folders:
        with open(log + '/settings', 'r') as f:
            settings = json.load(f)
            armotized = settings['armotized']

        log_base = os.path.basename(os.path.normpath(log))
        print ('iLQR for ' + log_base)

        model = PCC(armotized, 1600, 2, 2, 'planar').to(device)
        model.load_state_dict(torch.load(log + '/model_5000'))
        model.eval()
        dynamics = model.dynamics
        encoder = model.encoder

        # draw random initial state and goal state
        s_start = np.random.uniform(planar_sampler.rw, planar_sampler.rw + 4, size=2)
        s_goal = np.random.uniform(planar_sampler.height - planar_sampler.rw - 4,
                            planar_sampler.height - planar_sampler.rw, size = 2)
        image_start = planar_sampler.render(s_start)
        image_goal = planar_sampler.render(s_goal)
        x_start = torch.from_numpy(image_start).cuda().squeeze(0).view(-1, 1600)
        x_goal = torch.from_numpy(image_goal).cuda().squeeze(0).view(-1, 1600)

        z_start, _ = model.encode(x_start)
        z_goal, _ = model.encode(x_goal)
        u_opt = reciding_horizon(R_z, R_u, s_start, z_start, z_goal, dynamics, encoder, 10, 40)
        s = s_start
        traj_path = 'trajectory/' + log_base
        if not os.path.exists(traj_path):
            os.makedirs(traj_path)
        Image.fromarray(image_start * 255.).convert('L').save(traj_path + '/0.png')
        for i, u in enumerate(u_opt):
            if u is not None:
                u = np.array(u.squeeze().cpu().detach())
                s = s + u
                image = planar_sampler.render(s)
                Image.fromarray(image * 255.).convert('L').save(traj_path + '/' + str(i+1) + '.png')
        Image.fromarray(image_goal * 255.).convert('L').save(traj_path + '/goal.png')

if __name__ == '__main__':
    main()