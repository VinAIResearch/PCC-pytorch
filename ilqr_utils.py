import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from torchvision.transforms import ToTensor

np.random.seed(0)

def cost_dz(R_z, z, z_goal):
    # compute the first-order deravative of latent cost w.r.t z
    return 2 * R_z.mm((z - z_goal).view(-1,1))

def cost_du(R_u, u):
    # compute the first-order deravative of latent cost w.r.t u
    return 2 * R_u.mm(u.view(-1,1))

def cost_dzz(R_z):
    # compute the second-order deravative of latent cost w.r.t z
    return 2 * R_z

def cost_duu(R_u):
    # compute the second-order deravative of latent cost w.r.t u
    return 2 * R_u

def cost_duz():
    # compute the second-order deravative of latent cost w.r.t uz
    return 0

def latent_cost(R_z, R_u, z_seq, z_goal, u_seq):
    cost = 0.0
    for t in range(len(z_seq) - 1):
        z_t_diff, u_t = z_seq[t] - z_goal, u_seq[t]
        cost += z_t_diff.view(1,-1).mm(R_z).mm(z_t_diff.view(-1,1)) + u_t.view(1,-1).mm(R_u).mm(u_t.view(-1,1))
    z_T_diff = z_seq[-1] - z_goal
    cost += z_T_diff.view(1,-1).mm(R_z).mm(z_T_diff.view(-1,1))
    return cost

def one_step_back(R_z, R_u, z, u, z_goal, A, B, V_prime_next_z, V_prime_next_zz, mu_inv_regulator):
    """
    V_prime_next_z: first order derivative of the value function at time step t+1
    V_prime_next_zz: second order derivative of the value function at time tep t+1
    A: derivative of F(z, u) w.r.t z at z_bar_t, u_bar_t
    B: derivative of F(z, u) w.r.t u at z_bar_t, u_bar_t
    """
    # compute Q_z, Q_u, Q_zz, Q_uu, Q_uz using cost function, A, B and V
    Q_z = cost_dz(R_z, z, z_goal) + A.transpose(1,0).mm(V_prime_next_z.view(-1,1))
    Q_u = cost_du(R_u, u) + B.transpose(1,0).mm(V_prime_next_z.view(-1,1))
    Q_zz = cost_dzz(R_z) + A.transpose(1,0).mm(V_prime_next_zz).mm(A)
    Q_uz = cost_duz() + B.transpose(1,0).mm(V_prime_next_zz).mm(A)
    Q_uu = cost_duu(R_u) + B.transpose(1,0).mm(V_prime_next_zz).mm(B)

    # compute k and K matrix, add regularization to Q_uu
    Q_uu_regularized = Q_uu + mu_inv_regulator * torch.eye(Q_u.size(0)).cuda()
    Q_uu_in = torch.inverse(Q_uu_regularized)
    k = -Q_uu_in.mm(Q_u)
    K = -Q_uu_in.mm(Q_uz)

    # compute V_z and V_zz using k and K
    V_prime_z = Q_z.transpose(1,0) - k.transpose(1,0).mm(Q_uu).mm(K)
    V_prime_zz = Q_zz - K.transpose(1,0).mm(Q_uu).mm(K)
    return k, K, V_prime_z, V_prime_zz

def backward(R_z, R_u, z_seq, u_seq, z_goal, A_seq, B_seq, mu_inv_regulator):
    """
    do the backward pass
    return a sequence of k and K matrices
    """
    # first and second order derivative of the value function at time step T
    V_prime_next_z = cost_dz(R_z, z_seq[-1], z_goal)
    V_prime_next_zz = cost_dzz(R_z)
    k, K = [], []
    act_seq_len = len(z_seq)
    for i in range(2, act_seq_len + 1):
        # print ('Backward step ' + str(i-2))
        t = act_seq_len - i
        k_t, K_t, V_prime_z, V_prime_zz = one_step_back(R_z, R_u, z_seq[t], u_seq[t], z_goal, A_seq[t], B_seq[t], V_prime_next_z, V_prime_next_zz, mu_inv_regulator)
        k.insert(0, k_t)
        K.insert(0, K_t)
        V_prime_next_z, V_prime_next_zz = V_prime_z, V_prime_zz
    return k, K

def forward(u_seq, k, K, A_seq, B_seq, alpha):
    """
    update the trajectory, given k and K
    !!!! update using the linearization matricies (A and B), not the learned dynamics
    """
    u_new_seq = []
    horizon = len(u_seq)
    z_dim = K[0].size(1)
    for i in range(0, horizon):
        if i == 0:
            z_delta = torch.zeros(size=(z_dim, 1)).cuda()
        else:
            z_delta = A_seq[i-1].mm(z_delta) + B_seq[i-1].mm(u_delta)
        u_delta = alpha * k[i] + K[i].mm(z_delta)
        u_new_seq.append(u_seq[i] + u_delta.view(1,-1))
    return u_new_seq

def random_uniform_actions(mdp, horizon):
    # create a trajectory of random actions
    random_actions = []
    for i in range(horizon):
        action = mdp.sample_random_action()
        action = torch.from_numpy(action).cuda().view(1, -1).double()
        random_actions.append(action)
    return random_actions

def random_extreme_actions(mdp, horizon):
    # create a trajectory of extreme actions
    extreme_actions = []
    for i in range(horizon):
        action = mdp.sample_extreme_action()
        action = torch.from_numpy(action).cuda().view(1, -1).double()
        extreme_actions.append(action)
    return extreme_actions

def random_actions_trajs(mdp, num_uniform, num_extreme, horizon):
    actions_trajs = []
    for i in range(num_uniform):
        actions_trajs.append(random_uniform_actions(mdp, horizon))
    for j in range(num_extreme):
        actions_trajs.append(random_extreme_actions(mdp, horizon))
    return actions_trajs

def refresh_actions_trajs(actions_trajs, traj_opt_id, mdp, length, num_uniform, num_extreme):
    for traj_id in range(len(actions_trajs)):
        if traj_id == traj_opt_id:
            actions_trajs[traj_id] = actions_trajs[traj_id][1:]
            continue
        if traj_id < num_uniform:
            actions_trajs[traj_id] = random_uniform_actions(mdp, length)
        else:
            actions_trajs[traj_id] = random_extreme_actions(mdp, length)
    return actions_trajs

def compute_latent_traj(s_start, u_seq, env_name, mdp, dynamics, encoder):
    x_start = mdp.render(s_start).squeeze()
    if env_name == 'pendulum':
        x_start = np.vstack((x_start, x_start))
    x_start = ToTensor()(x_start).double().cuda().view(-1, encoder.x_dim)
    with torch.no_grad():
        z_start, _ = encoder(x_start)
    horizon = len(u_seq)
    z_seq = [z_start]
    for i in range(horizon):
        z = z_seq[i]
        u = u_seq[i]
        with torch.no_grad():
            z_next, _, _, _ = dynamics(z, u)
        z_seq.append(z_next)
    return z_seq

def jacobian(dynamics, z, u):
    """
    compute the jacobian of F(z,u) w.r.t z, u
    """
    z_dim = z.size(1)
    u_dim = u.size(1)
    if dynamics.armotized:
        z_next, _, A, B = dynamics(z, u)
        return A.view(z_dim, z_dim), B.view(z_dim, u_dim)
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
    horizon = len(u_seq)
    for i in range(horizon):
        z, u = z_seq[i], u_seq[i]
        A, B = jacobian(dynamics, z, u)
        A_seq.append(A)
        B_seq.append(B)
    return A_seq, B_seq

def random_start_goal(env_name, mdp):
    """
    return a random start state and the goal state
    """
    if env_name == 'planar':
        s_goal = np.random.uniform(mdp.height - mdp.half_agent_size - 3,
                                    mdp.width - mdp.half_agent_size, size=2)
        idx = np.random.randint(0, 3)
        if idx == 0:
            s_start = np.random.uniform(mdp.half_agent_size, mdp.half_agent_size + 3, size=2)
        if idx == 1:
            x = np.random.uniform(mdp.half_agent_size, mdp.half_agent_size+3)
            y = np.random.uniform(mdp.width - mdp.half_agent_size - 3,
                                mdp.width - mdp.half_agent_size)
            s_start = np.array([x, y])
        if idx == 2:
            x = np.random.uniform(mdp.width - mdp.half_agent_size - 3,
                                mdp.width - mdp.half_agent_size)
            y = np.random.uniform(mdp.half_agent_size, mdp.half_agent_size+3)
            s_start = np.array([x, y])
    elif env_name == 'pendulum':
        s_goal = np.zeros(2)
        idx = np.random.randint(0,2)
        if idx == 0: # swing up
            s_start = np.array([np.pi, np.random.uniform(mdp.angular_rate_limits[0],
                                       mdp.angular_rate_limits[1])])
        if idx == 1: # balance
            s_start = np.array([0.0, np.random.uniform(mdp.angular_rate_limits[0],
                                                         mdp.angular_rate_limits[1])])
    return idx, s_start, s_goal

def save_traj(images, image_goal, gif_path, env_name):
    # save trajectory as gif file
    fig, aa = plt.subplots(1, 2)
    m1 = aa[0].matshow(
        images[0], cmap=plt.cm.gray, vmin=0., vmax=1.)
    aa[0].set_title('Time step 0')
    aa[0].set_yticklabels([])
    aa[0].set_xticklabels([])
    m2 = aa[1].matshow(
        image_goal, cmap=plt.cm.gray, vmin=0., vmax=1.)
    aa[1].set_title('goal')
    aa[1].set_yticklabels([])
    aa[1].set_xticklabels([])
    fig.tight_layout()

    def updatemat2(t):
        m1.set_data(images[t])
        aa[0].set_title('Time step ' + str(t))
        m2.set_data(image_goal)
        return m1, m2

    if env_name == 'planar':
        anim = FuncAnimation(
            fig, updatemat2, frames=40, interval=200, blit=True, repeat=True)
        Writer = writers['imagemagick']  # animation.writers.avail
        writer = Writer(fps=4, metadata=dict(artist='Me'), bitrate=1800)
    elif env_name == 'pendulum':
        anim = FuncAnimation(
            fig, updatemat2, frames=100, interval=200, blit=True, repeat=True)
        Writer = writers['imagemagick']  # animation.writers.avail
        writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)

    anim.save(gif_path, writer=writer)