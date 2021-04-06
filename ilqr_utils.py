import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, writers


np.random.seed(0)


def cost_dz(R_z, z, z_goal):
    # compute the first-order deravative of latent cost w.r.t z
    z_diff = np.expand_dims(z - z_goal, axis=-1)
    return np.squeeze(2 * np.matmul(R_z, z_diff))


def cost_du(R_u, u):
    # compute the first-order deravative of latent cost w.r.t u
    return np.atleast_1d(np.squeeze(2 * np.matmul(R_u, np.expand_dims(u, axis=-1))))


def cost_dzz(R_z):
    # compute the second-order deravative of latent cost w.r.t z
    return 2 * R_z


def cost_duu(R_u):
    # compute the second-order deravative of latent cost w.r.t u
    return 2 * R_u


def cost_duz(z, u):
    # compute the second-order deravative of latent cost w.r.t uz
    return np.zeros((u.shape[-1], z.shape[-1]))


def latent_cost(R_z, R_u, z_seq, z_goal, u_seq):
    z_diff = np.expand_dims(z_seq - z_goal, axis=-1)
    cost_z = np.squeeze(np.matmul(np.matmul(z_diff.transpose((0, 2, 1)), R_z), z_diff))
    u_seq_reshaped = np.expand_dims(u_seq, axis=-1)
    cost_u = np.squeeze(np.matmul(np.matmul(u_seq_reshaped.transpose((0, 2, 1)), R_u), u_seq_reshaped))
    return np.sum(cost_z) + np.sum(cost_u)


def one_step_back(R_z, R_u, z, u, z_goal, A, B, V_prime_next_z, V_prime_next_zz, mu_inv_regulator):
    """
    V_prime_next_z: first order derivative of the value function at time step t+1
    V_prime_next_zz: second order derivative of the value function at time tep t+1
    A: derivative of F(z, u) w.r.t z at z_bar_t, u_bar_t
    B: derivative of F(z, u) w.r.t u at z_bar_t, u_bar_t
    """
    # compute Q_z, Q_u, Q_zz, Q_uu, Q_uz using cost function, A, B and V
    Q_z = cost_dz(R_z, z, z_goal) + np.matmul(A.transpose(), V_prime_next_z)
    Q_u = cost_du(R_u, u) + np.matmul(B.transpose(), V_prime_next_z)
    Q_zz = cost_dzz(R_z) + np.matmul(np.matmul(A.transpose(), V_prime_next_zz), A)
    Q_uz = cost_duz(z, u) + np.matmul(np.matmul(B.transpose(), V_prime_next_zz), A)
    Q_uu = cost_duu(R_u) + np.matmul(np.matmul(B.transpose(), V_prime_next_zz), B)

    # compute k and K matrix, add regularization to Q_uu
    Q_uu_regularized = Q_uu + mu_inv_regulator * np.eye(Q_uu.shape[0])
    Q_uu_in = np.linalg.inv(Q_uu_regularized)
    k = -np.matmul(Q_uu_in, Q_u)
    K = -np.matmul(Q_uu_in, Q_uz)

    # compute V_z and V_zz using k and K
    V_prime_z = Q_z + np.matmul(Q_uz.transpose(), k)
    V_prime_zz = Q_zz + np.matmul(Q_uz.transpose(), K)
    return k, K, V_prime_z, V_prime_zz


def backward(R_z, R_u, z_seq, u_seq, z_goal, A_seq, B_seq, inv_regulator):
    """
    do the backward pass
    return a sequence of k and K matrices
    """
    # first and second order derivative of the value function at the last time step
    V_prime_next_z = cost_dz(R_z, z_seq[-1], z_goal)
    V_prime_next_zz = cost_dzz(R_z)
    k, K = [], []
    act_seq_len = len(u_seq)
    for t in reversed(range(act_seq_len)):
        k_t, K_t, V_prime_z, V_prime_zz = one_step_back(
            R_z, R_u, z_seq[t], u_seq[t], z_goal, A_seq[t], B_seq[t], V_prime_next_z, V_prime_next_zz, inv_regulator
        )
        k.insert(0, k_t)
        K.insert(0, K_t)
        V_prime_next_z, V_prime_next_zz = V_prime_z, V_prime_zz
    return k, K


def forward(z_seq, u_seq, k, K, dynamics, alpha):
    """
    update the trajectory, given k and K
    !!!! update using the linearization matricies (A and B), not the learned dynamics
    """
    z_seq_new = []
    z_seq_new.append(z_seq[0])
    u_seq_new = []
    for i in range(0, len(u_seq)):
        u_new = u_seq[i] + alpha * k[i] + np.matmul(K[i], z_seq_new[i] - z_seq[i])
        u_seq_new.append(u_new)
        with torch.no_grad():
            z_new = dynamics(torch.from_numpy(z_seq_new[i]).unsqueeze(0), torch.from_numpy(u_new).unsqueeze(0))[0].mean
        z_seq_new.append(z_new.squeeze().numpy())
    return np.array(z_seq_new), np.array(u_seq_new)


# def forward(u_seq, k_seq, K_seq, A_seq, B_seq, alpha):
#     """
#     update the trajectory, given k and K
#     !!!! update using the linearization matricies (A and B), not the learned dynamics
#     """
#     u_new_seq = []
#     plan_len = len(u_seq)
#     z_dim = K_seq[0].shape[1]
#     for i in range(0, plan_len):
#         if i == 0:
#             z_delta = np.zeros(z_dim)
#         else:
#             z_delta = np.matmul(A_seq[i-1], z_delta) + np.matmul(B_seq[i-1], u_delta)
#         u_delta = alpha * (k_seq[i] + np.matmul(K_seq[i], z_delta))
#         u_new_seq.append(u_seq[i] + u_delta)
#     return np.array(u_new_seq)


def get_x_data(mdp, state, config):
    image_data = mdp.render(state).squeeze()
    x_dim = config["obs_shape"]
    if config["task"] == "plane":
        x_dim = np.prod(x_dim)
        x_data = torch.from_numpy(image_data).double().view(x_dim).unsqueeze(0)
    elif config["task"] in ["swing", "balance"]:
        x_dim = np.prod(x_dim)
        x_data = np.vstack((image_data, image_data))
        x_data = torch.from_numpy(x_data).double().view(x_dim).unsqueeze(0)
    elif config["task"] in ["cartpole", "threepole"]:
        x_data = torch.zeros(size=(2, 80, 80))
        x_data[0, :, :] = torch.from_numpy(image_data)
        x_data[1, :, :] = torch.from_numpy(image_data)
        x_data = x_data.unsqueeze(0)
    return x_data


def update_horizon_start(mdp, s, u, encoder, config):
    s_next = mdp.transition_function(s, u)
    if config["task"] == "plane":
        x_next = get_x_data(mdp, s_next, config)
    elif config["task"] in ["swing", "balance"]:
        obs = mdp.render(s).squeeze()
        obs_next = mdp.render(s_next).squeeze()
        obs_stacked = np.vstack((obs, obs_next))
        x_dim = np.prod(config["obs_shape"])
        x_next = torch.from_numpy(obs_stacked).view(x_dim).unsqueeze(0).double()
    elif config["task"] in ["cartpole", "threepole"]:
        obs = mdp.render(s).squeeze()
        obs_next = mdp.render(s_next).squeeze()
        x_next = torch.zeros(size=config["obs_shape"])
        x_next[0, :, :] = torch.from_numpy(obs)
        x_next[1, :, :] = torch.from_numpy(obs_next)
        x_next = x_next.unsqueeze(0)
    with torch.no_grad():
        z_next = encoder(x_next).mean
    return s_next, z_next.squeeze().numpy()


def random_uniform_actions(mdp, plan_len):
    # create a trajectory of random actions
    random_actions = []
    for i in range(plan_len):
        action = mdp.sample_random_action()
        random_actions.append(action)
    return np.array(random_actions)


def random_extreme_actions(mdp, plan_len):
    # create a trajectory of extreme actions
    extreme_actions = []
    for i in range(plan_len):
        action = mdp.sample_extreme_action()
        extreme_actions.append(action)
    return np.array(extreme_actions)


def random_actions_trajs(mdp, num_uniform, num_extreme, plan_len):
    actions_trajs = []
    for i in range(num_uniform):
        actions_trajs.append(random_uniform_actions(mdp, plan_len))
    for j in range(num_extreme):
        actions_trajs.append(random_extreme_actions(mdp, plan_len))
    return actions_trajs


def refresh_actions_trajs(actions_trajs, traj_opt_id, mdp, length, num_uniform, num_extreme):
    for traj_id in range(len(actions_trajs)):
        if traj_id == traj_opt_id:
            actions_trajs[traj_id] = actions_trajs[traj_id][1:]
            if len(actions_trajs[traj_id]) < length:
                # Duplicate last action.
                actions_trajs[traj_id] = np.append(
                    actions_trajs[traj_id], actions_trajs[traj_id][-1].reshape(1, -1), axis=0
                )
            continue
        if traj_id < num_uniform:
            actions_trajs[traj_id] = random_uniform_actions(mdp, length)
        else:
            actions_trajs[traj_id] = random_extreme_actions(mdp, length)
    return actions_trajs


def update_seq_act(z_seq, z_start, u_seq, k, K, dynamics):
    """
    update the trajectory, given k and K
    """
    z_new = z_start
    u_seq_new = []
    for i in range(0, len(u_seq)):
        u_new = u_seq[i] + k[i] + np.matmul(K[i], (z_new - z_seq[i]))
        with torch.no_grad():
            z_new = dynamics(torch.from_numpy(z_new).view(1, -1), torch.from_numpy(u_new).view(1, -1))[0].mean
            z_new = z_new.squeeze().numpy()
        u_seq_new.append(u_new)
    return np.array(u_seq_new)


def compute_latent_traj(z_start, u_seq, dynamics):
    plan_len = len(u_seq)
    z_seq = [z_start]
    for i in range(plan_len):
        z = torch.from_numpy(z_seq[i]).view(1, -1).double()
        u = torch.from_numpy(u_seq[i]).view(1, -1).double()
        with torch.no_grad():
            z_next = dynamics(z, u)[0].mean
        z_seq.append(z_next.squeeze().numpy())
    return z_seq


def jacobian(dynamics, z, u):
    """
    compute the jacobian of F(z,u) w.r.t z, u
    """
    z_dim = z.shape[0]
    u_dim = u.shape[0]
    z_tensor = torch.from_numpy(z).view(1, -1).double()
    u_tensor = torch.from_numpy(u).view(1, -1).double()
    if dynamics.armotized:
        _, A, B = dynamics(z_tensor, u_tensor)
        return A.squeeze().view(z_dim, z_dim).numpy(), B.squeeze().view(z_dim, u_dim).numpy()
    z_tensor, u_tensor = z_tensor.squeeze().repeat(z_dim, 1), u_tensor.squeeze().repeat(z_dim, 1)
    z_tensor = z_tensor.detach().requires_grad_(True)
    u_tensor = u_tensor.detach().requires_grad_(True)
    z_next = dynamics(z_tensor, u_tensor)[0].mean
    grad_inp = torch.eye(z_dim)
    A, B = torch.autograd.grad(z_next, [z_tensor, u_tensor], [grad_inp, grad_inp])
    return A.numpy(), B.numpy()


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


def save_traj(images, image_goal, gif_path, task):
    # save trajectory as gif file
    fig, aa = plt.subplots(1, 2)
    m1 = aa[0].matshow(images[0], cmap=plt.cm.gray, vmin=0.0, vmax=1.0)
    aa[0].set_title("Time step 0")
    aa[0].set_yticklabels([])
    aa[0].set_xticklabels([])
    m2 = aa[1].matshow(image_goal, cmap=plt.cm.gray, vmin=0.0, vmax=1.0)
    aa[1].set_title("goal")
    aa[1].set_yticklabels([])
    aa[1].set_xticklabels([])
    fig.tight_layout()

    def updatemat2(t):
        m1.set_data(images[t])
        aa[0].set_title("Time step " + str(t))
        m2.set_data(image_goal)
        return m1, m2

    frames = len(images)
    if task == "plane":
        fps = 2
    else:
        fps = 20

    anim = FuncAnimation(fig, updatemat2, frames=frames, interval=200, blit=True, repeat=True)
    Writer = writers["imagemagick"]  # animation.writers.avail
    writer = Writer(fps=fps, metadata=dict(artist="Me"), bitrate=1800)

    anim.save(gif_path, writer=writer)
