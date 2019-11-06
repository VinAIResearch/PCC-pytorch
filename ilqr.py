import argparse

from pcc_model import PCC
import data.sample_planar as planar_sampler
from mdp.plane_obstacles_mdp import PlanarObstaclesMDP
from mdp.pole_simple_mdp import VisualPoleSimpleSwingUp
from ilqr_utils import *
from datasets import *

np.random.seed(0)
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

mdps = {'planar': PlanarObstaclesMDP, 'pendulum': VisualPoleSimpleSwingUp}
network_dims = {'planar': (1600, 2, 2), 'pendulum': (4608, 3, 1)}
img_dims = {'planar': (40, 40), 'pendulum': (48, 96)}
horizons = {'planar': 40, 'pendulum': 400}
R_z = 10
R_u = 1


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


def compute_loss(R_z, R_u, z_seq, z_goal, u_seq):
    z_diff = np.expand_dims(z_seq - z_goal, axis=-1)
    cost_z = np.squeeze(np.matmul(
        np.matmul(z_diff.transpose((0, 2, 1)), R_z), z_diff))
    u_seq_reshaped = np.expand_dims(u_seq, axis=-1)
    cost_u = np.squeeze(np.matmul(
        np.matmul(u_seq_reshaped.transpose((0, 2, 1)), R_u), u_seq_reshaped))
    return np.sum(cost_z) + np.sum(cost_u)


def one_step_back(R_z, R_u, z, u, z_goal, A, B, V_prime_next_z, V_prime_next_zz):
    """
    V_next_z: first order derivative of the value function at time step t+1
    V_next_zz: second order derivative of the value function at time tep t+1
    A: derivative of F(z, u) w.r.t z at z_bar_t, u_bar_t
    B: derivative of F(z, u) w.r.t u at z_bar_t, u_bar_t
    """
    # compute Q_z, Q_u, Q_zz, Q_uu, Q_uz using cost function, A, B and V
    Q_z = cost_dz(R_z, z, z_goal) + np.matmul(A.transpose(), V_prime_next_z)
    Q_u = cost_du(R_u, u) + np.matmul(B.transpose(), V_prime_next_z)
    Q_zz = cost_dzz(R_z) + np.matmul(np.matmul(A.transpose(), V_prime_next_zz), A)
    Q_uz = cost_duz(z, u) + np.matmul(np.matmul(B.transpose(), V_prime_next_zz), A)
    Q_uu = cost_duu(R_u) + np.matmul(np.matmul(B.transpose(), V_prime_next_zz), B)

    # compute k and K matrix
    Q_uu_in = np.linalg.inv(Q_uu)
    k = -np.matmul(Q_uu_in, Q_u)
    K = -np.matmul(Q_uu_in, Q_uz)
    # compute V_z and V_zz using k and K
    V_prime_z = Q_z + np.matmul(Q_uz.transpose(), k)
    V_prime_zz = Q_zz + np.matmul(Q_uz.transpose(), K)
    return k, K, V_prime_z, V_prime_zz


def backward(R_z, R_u, z_seq, u_seq, z_goal, A_seq, B_seq):
    """
    do the backward pass
    return a sequence of k and K matrices
    """
    V_prime_next_z = cost_dz(R_z, z_seq[-1], z_goal)
    V_prime_next_zz = cost_dzz(R_z)
    k, K = [], []
    act_seq_len = len(u_seq)
    for t in reversed(range(act_seq_len)):
        k_t, K_t, V_prime_z, V_prime_zz = one_step_back(R_z, R_u, z_seq[t], u_seq[t], z_goal, A_seq[t], B_seq[t],
                                                        V_prime_next_z, V_prime_next_zz)
        k.insert(0, k_t)
        K.insert(0, K_t)
        V_prime_next_z, V_prime_next_zz = V_prime_z, V_prime_zz
    return k, K


def forward(z_seq, u_seq, k, K, dynamics):
    """
    update the trajectory, given k and K
    !!!! update using the linearization matricies (A and B), not the learned dynamics
    """
    z_seq_new = []
    z_seq_new.append(z_seq[0])
    u_seq_new = []
    for i in range(0, len(u_seq)):
        u_new = u_seq[i] + k[i] + np.matmul(K[i], z_seq_new[i] - z_seq[i])
        u_seq_new.append(u_new)
        with torch.no_grad():
            z_new, _, _, _ = dynamics(torch.from_numpy(z_seq_new[i]).unsqueeze(0),
                                      torch.from_numpy(u_new).unsqueeze(0))
        z_seq_new.append(z_new.squeeze().numpy())
    return z_seq_new, u_seq_new


def iLQR_solver(R_z, R_u, z_seq, z_goal, u_seq, dynamics, iters):
    """
    - run backward: linearize around the current trajectory and perform optimal control
    - run forward: update the current trajectory
    - repeat
    """
    loss = compute_loss(R_z, R_u, z_seq, z_goal, u_seq)
    A_seq, B_seq = seq_jacobian(dynamics, z_seq, u_seq)
    print('iLQR loss iter {:02d}: {:05f}'.format(0, loss.item()))
    for i in range(iters):
        k, K = backward(R_z, R_u, z_seq, u_seq, z_goal, A_seq, B_seq)
        z_seq, u_seq = forward(z_seq, u_seq, k, K, dynamics)
        loss = compute_loss(R_z, R_u, z_seq, z_goal, u_seq)
        print('iLQR loss iter {:02d}: {:05f}'.format(i + 1, loss.item()))
        # print ('iLQR step ' + str(i))
        A_seq, B_seq = seq_jacobian(dynamics, z_seq, u_seq)
    return z_seq, u_seq, k, K


def update_seq_act(z_seq, z_start, u_seq, k, K, dynamics):
    """
    update the trajectory, given k and K
    """
    z_seq_new = [z_start.squeeze().numpy()]
    u_seq_new = []
    for i in range(0, len(u_seq)):
        u_new = u_seq[i] + k[i] + np.matmul(K[i], (z_seq_new[i] - z_seq[i]))
        with torch.no_grad():
            z_new, _, _, _ = dynamics(torch.from_numpy(z_seq_new[i]).view(1, -1),
                                      torch.from_numpy(u_new).view(1, -1))
        u_seq_new.append(u_new)
        z_seq_new.append(z_new.squeeze().numpy())
    return np.array(z_seq_new), np.array(u_seq_new)


def reciding_horizon(env_name, mdp, x_dim, R_z, R_u, s_start, z_start, z_goal, dynamics, encoder, iters_ilqr, horizon):
    # for the first step
    z_seq, u_seq = random_traj(env_name, mdp, s_start, z_start, horizon, dynamics)
    u_opt = []
    s = s_start
    for i in range(horizon):
        # optimal perturbed policy at time step t
        print('Horizon {:02d}'.format(i + 1))
        z_seq, u_seq, k, K = iLQR_solver(R_z, R_u, z_seq, z_goal, u_seq, dynamics, iters_ilqr)
        u_first_opt = u_seq[0]
        if np.any(np.isnan(u_first_opt)):
            return None
        u_opt.append(u_first_opt)

        # get z_t+1 from the true dynamics
        if env_name == 'planar':
            s = mdp.transition_function(s, u_first_opt)
            next_obs = mdp.render(s)
            next_x = torch.from_numpy(next_obs).unsqueeze(0).view(-1, x_dim)
        elif env_name == 'pendulum':
            image = mdp.render(s).squeeze()
            s, next_image = mdp.transition_function((s, image), u_first_opt)
            next_obs = np.hstack((image, next_image.squeeze()))
            next_x = Image.fromarray(next_obs * 255.).convert('L')
            next_x = ToTensor()(next_x.convert('L').resize((96, 48))).double()
            next_x = torch.cat((next_x[:, :, :48], next_x[:, :, 48:]), dim=1).view(-1, x_dim)
        with torch.no_grad():
            z_start, _ = encoder(next_x)

        # update the nominal trajectory
        z_seq, u_seq = z_seq[1:], u_seq[1:]
        k, K = k[1:], K[1:]
        z_seq, u_seq = update_seq_act(z_seq, z_start, u_seq, k, K, dynamics)
        print('==============================')
        # loss = compute_loss(R_z, R_u, z_seq, z_goal, u_seq)
        # if torch.isnan(loss):
        #     return None
    return u_opt


def main(args):
    env_name = args.env
    ilqr_iters = args.ilqr_iters
    mdp = mdps[env_name]()
    x_dim, z_dim, u_dim = network_dims[env_name]
    horizon = horizons[env_name]
    R_z = 10 * np.eye(z_dim)
    R_u = 1 * np.eye(u_dim)

    folder = 'result/' + env_name
    log_folders = [os.path.join(folder, dI) for dI in os.listdir(folder) if os.path.isdir(os.path.join(folder, dI))]
    log_folders.sort()

    avg_model_percent = 0.0
    best_model_percent = 0.0
    for log in log_folders:
        with open(log + '/settings', 'r') as f:
            settings = json.load(f)
            armotized = settings['armotized']

        log_base = os.path.basename(os.path.normpath(log))
        model_path = 'iLQR_result/' + env_name + '/' + log_base
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        print('Performing iLQR for ' + log_base)

        model = PCC(armotized, x_dim, z_dim, u_dim, env_name)
        model.load_state_dict(torch.load(log + '/model_5000'))
        model.eval()
        dynamics = model.dynamics
        encoder = model.encoder

        avg_percent = 0
        for task in range(10):
            print('Performing task ' + str(task + 1))
            # draw random initial state and goal state
            idx, s_start, image_start, s_goal, image_goal = random_start_goal(env_name, mdp)
            if env_name == 'planar':
                x_start = torch.from_numpy(image_start).unsqueeze(0).view(-1, x_dim)
                x_goal = torch.from_numpy(image_goal).unsqueeze(0).view(-1, x_dim)
            elif env_name == 'pendulum':
                x_start = Image.fromarray(image_start * 255.).convert('L')
                x_start = ToTensor()(x_start.convert('L').resize((96, 48))).double()
                x_start = torch.cat((x_start[:, :, :48], x_start[:, :, 48:]), dim=1).view(-1, x_dim)

                x_goal = Image.fromarray(image_goal * 255.).convert('L')
                x_goal = ToTensor()(x_goal.convert('L').resize((96, 48))).double()
                x_goal = torch.cat((x_goal[:, :, :48], x_goal[:, :, 48:]), dim=1).view(-1, x_dim)
            with torch.no_grad():
                z_start, _ = model.encode(x_start)
                z_goal, _ = model.encode(x_goal)
            z_start = z_start.squeeze().numpy()
            z_goal = z_goal.squeeze().numpy()

            # perform optimal control for this task
            u_opt = reciding_horizon(env_name, mdp, x_dim, R_z, R_u, s_start, z_start, z_goal, dynamics, encoder,
                                     ilqr_iters, horizon)
            if u_opt is None:
                avg_percent += 0
                with open(model_path + '/result.txt', 'a+') as f:
                    f.write('Task {:01d} start at corner {:01d}: '.format(task + 1, idx) + ' crashed' + '\n')
                continue

            # compute the trajectory
            s = s_start
            if env_name != 'planar':
                image_start = image_start[:, :48]
                image_goal = image_goal[:, :48]
            images = [image_start]
            reward = 0.0
            for i, u in enumerate(u_opt):
                u = u.squeeze()
                if env_name == 'planar':
                    s = mdp.transition_function(s, u)
                    image = mdp.render(s)
                    images.append(image)
                    reward += mdp.reward_function(s, s_goal)
                elif env_name == 'pendulum':
                    s, image = mdp.transition_function((s, images[-1]), u)
                    images.append(image.squeeze())
                    reward += mdp.reward_function((s, image))

            # compute the percentage close to goal
            percent = reward / horizon
            avg_percent += percent
            with open(model_path + '/result.txt', 'a+') as f:
                f.write('Task {:01d} start at corner {:01d}: '.format(task + 1, idx) + str(percent) + '\n')

            # save trajectory as gif file
            gif_path = model_path + '/task_{:01d}.gif'.format(task + 1)
            save_traj(images, image_goal, gif_path, env_name)

        avg_percent = avg_percent / 10
        avg_model_percent += avg_percent
        if avg_percent > best_model_percent:
            best_model = log_base
            best_model_percent = avg_percent
        with open(model_path + '/result.txt', 'a+') as f:
            f.write('Average percentage: ' + str(avg_percent))

    avg_model_percent = avg_model_percent / len(log_folders)
    with open('iLQR_result/result.txt', 'w') as f:
        f.write('Average percentage of all models: ' + str(avg_model_percent) + '\n')
        f.write('Best model: ' + best_model + ', best percentage: ' + str(best_model_percent))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train pcc model')
    parser.add_argument('--env', required=True, type=str, help='name of the environment')
    parser.add_argument('--ilqr_iters', required=True, type=int, help='the number of ilqr iterations')

    args = parser.parse_args()

    main(args)