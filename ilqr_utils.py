import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
import numpy as np
import torch
from mdp.plane_obstacles_mdp import PlanarObstaclesMDP

np.random.seed(0)

def random_traj(mdp, s_start, z_start, horizon, dynamics):
    """
    initialize a valid random trajectory
    """
    z = z_start
    z_seq = []
    u_seq = []
    s = s_start
    for i in range(horizon):
        z_seq.append(z)
        u = mdp.sample_valid_random_action(s)
        u = torch.from_numpy(u).view(1,-1).cuda()
        u_seq.append(u)
        with torch.no_grad():
            z_next, _, _, _ = dynamics(z, u)
        z = z_next
        s = mdp.transition_function(s, u.cpu().detach().numpy())
    z_seq.append(z)
    u_seq.append(None)
    return z_seq, u_seq

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
    for i in range(len(z_seq) - 1):
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
        s_goal = np.array([0, np.random.uniform(-mdp.max_speed, mdp.max_speed)])
        idx = np.random.randint(0,2)
        if idx == 0: # swing up
            s_start = np.array([np.pi, np.random.uniform(-mdp.max_speed, mdp.max_speed)])
        if idx == 1: # balance
            theta_0 = np.random.uniform(-np.pi/6, np.pi/6)
            s_start = np.array([theta_0, np.random.uniform(-mdp.max_speed, mdp.max_speed)])
    return idx, s_start, s_goal

def angle_normalize(x):
    # for normalizing theta in pendulum
    return (((x+np.pi) % (2*np.pi)) - np.pi)

def is_close_goal(env_name, s, s_goal):
    # check if a state s is close to goal state
    if env_name == 'planar':
        return np.sqrt(np.sum(s - s_goal)**2) <= 2
    elif env_name == 'pendulum':
        theta = angle_normalize(s[0])
        return np.abs(theta) <= (np.pi / 6)

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
            fig, updatemat2, frames=400, interval=200, blit=True, repeat=True)
        Writer = writers['imagemagick']  # animation.writers.avail
        writer = Writer(fps=40, metadata=dict(artist='Me'), bitrate=1800)

    anim.save(gif_path, writer=writer)