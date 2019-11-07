import argparse
from torchvision.transforms import ToTensor
import torch

from pcc_model import PCC
from mdp.plane_obstacles_mdp import PlanarObstaclesMDP
from mdp.pole_simple_mdp import VisualPoleSimpleSwingUp
from ilqr_improved_utils import *
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

def main(args):
    env_name = args.env
    ilqr_iters = args.ilqr_iters
    num_uniform = args.num_uniform
    num_extreme = args.num_extreme
    init_mu = args.init_mu
    mu_mul = args.mu_mul
    mu_max = args.mu_max
    init_alpha = args.init_alpha
    alpha_mul = args.alpha_mul
    alpha_min = args.alpha_min

    mdp = mdps[env_name]()
    x_dim, z_dim, u_dim = network_dims[env_name]
    horizon = horizons[env_name]
    R_z = 10 * np.eye(z_dim)
    R_u = 1 * np.eye(u_dim)

    # get all models trained for this env
    folder = 'result/' + env_name
    log_folders = [os.path.join(folder, dI) for dI in os.listdir(folder) if os.path.isdir(os.path.join(folder, dI))]
    log_folders.sort()

    avg_model_percent = 0.0
    best_model_percent = 0.0
    # run iLQR for all models and compute the average performance
    for log in log_folders:
        with open(log + '/settings', 'r') as f:
            settings = json.load(f)
            armotized = settings['armotized']

        log_base = os.path.basename(os.path.normpath(log))
        model_path = 'iLQR_improved_result/' + env_name + '/' + log_base
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        print('iLQR for ' + log_base)

        # load the trained model
        model = PCC(armotized, x_dim, z_dim, u_dim, env_name)
        model.load_state_dict(torch.load(log + '/model_5000'))
        model.eval()
        dynamics = model.dynamics
        encoder = model.encoder

        # run iLQR for a particular model
        avg_percent = 0.0
        for task in range(10):  # perform 10 random tasks (10 different start states and goal states)
            print('Performing task ' + str(task + 1))
            # draw random initial state and goal state
            idx, s_start, s_goal = random_start_goal(env_name, mdp)
            image_start = mdp.render(s_start).squeeze()
            image_goal = mdp.render(s_goal).squeeze()
            if env_name == 'pendulum':
                image_goal = np.hstack((image_goal, image_goal))
                image_goal = Image.fromarray(image_goal * 255.).convert('L')
                image_goal = ToTensor()(image_goal.convert('L').resize((96, 48))).double()
                x_goal = torch.cat((image_goal[:, :, :48], image_goal[:, :, 48:]), dim=1).view(-1, x_dim)
            else:
                x_goal = ToTensor()(image_goal).double().view(-1, x_dim)
            with torch.no_grad():
                z_goal, _ = encoder(x_goal)

            z_goal = z_goal.squeeze().numpy()
            # initialize actions trajectories
            all_actions_trajs = random_actions_trajs(mdp, num_uniform, num_extreme, horizon)
            actions_final = []
            obs_traj = [image_start]
            goal_counter = 0.0

            # perform reciding horizon iLQR
            for plan_iter in range(1, horizon + 1):
                print('Planining iteration ' + str(plan_iter))
                latent_cost_list = [None] * len(all_actions_trajs)
                # iterate over all trajectories
                for traj_id in range(len(all_actions_trajs)):
                    print('Running iLQR for trajectory ' + str(traj_id + 1))
                    # initialize the inverse regulator
                    mu_inv_regulator = init_mu
                    for iter in range(1, ilqr_iters + 1):
                        # compute the latent trajectory
                        z_seq = compute_latent_traj(s_start, all_actions_trajs[traj_id],
                                                    env_name, mdp, dynamics, encoder)
                        # compute the linearization matrices
                        A_seq, B_seq = seq_jacobian(dynamics, z_seq, all_actions_trajs[traj_id])
                        # run backward
                        k_small, K_big = backward(R_z, R_u, z_seq, all_actions_trajs[traj_id],
                                                  z_goal, A_seq, B_seq, mu_inv_regulator)
                        # compute the latent cost for current u_seq
                        current_cost = latent_cost(R_z, R_u, z_seq, z_goal, all_actions_trajs[traj_id])
                        latent_cost_list[traj_id] = current_cost
                        # forward using line search
                        alpha = init_alpha
                        accept = False  # if any alpha is accepted
                        while alpha > alpha_min:
                            u_seq_cand = forward(all_actions_trajs[traj_id], k_small, K_big, A_seq, B_seq, alpha)
                            z_seq_cand = compute_latent_traj(s_start, u_seq_cand, env_name, mdp, dynamics, encoder)
                            cost_cand = latent_cost(R_z, R_u, z_seq_cand, z_goal, u_seq_cand)
                            if cost_cand < current_cost:  # accept the trajectory candidate
                                accept = True
                                all_actions_trajs[traj_id] = u_seq_cand
                                latent_cost_list[traj_id] = cost_cand
                                print('Found a better action sequence. New latent cost: ' + str(cost_cand.item()))
                                break
                            else:
                                alpha *= alpha_mul
                        if accept:
                            mu_inv_regulator = init_mu
                        else:
                            mu_inv_regulator *= mu_mul
                        if mu_inv_regulator > mu_max:
                            print('Can not improve. Stop iLQR.')
                            break

                    print('===================================================================')

                traj_opt_id = np.argmin(latent_cost_list)
                action_chosen = all_actions_trajs[traj_opt_id][0]
                actions_final.append(action_chosen)
                all_actions_trajs = refresh_actions_trajs(all_actions_trajs, traj_opt_id, mdp,
                                                          horizon - plan_iter, num_uniform, num_extreme)
                if env_name == 'planar':
                    goal_counter += mdp.reward_function(s_start, s_goal)
                    s_start = mdp.transition_function(s_start, action_chosen)
                    obs_traj.append(mdp.render(s_start).squeeze())
                else:
                    goal_counter += mdp.reward_function((s_start, obs_traj[-1]))
                    s_start, observation = mdp.transition_function((s_start, obs_traj[-1]), action_chosen)
                    obs_traj.append(observation.squeeze())

            # compute the percentage close to goal
            percent = goal_counter / horizon
            avg_percent += percent
            with open(model_path + '/result.txt', 'a+') as f:
                f.write('Task {:01d} start at corner {:01d}: '.format(task + 1, idx) + str(percent) + '\n')

            # save trajectory as gif file
            gif_path = model_path + '/task_{:01d}.gif'.format(task + 1)
            save_traj(obs_traj, image_goal.squeeze().numpy(), gif_path, env_name)

        avg_percent = avg_percent / 10
        avg_model_percent += avg_percent
        if avg_percent > best_model_percent:
            best_model = log_base
            best_model_percent = avg_percent
        with open(model_path + '/result.txt', 'a+') as f:
            f.write('Average percentage: ' + str(avg_percent))

    avg_model_percent = avg_model_percent / len(log_folders)
    with open('iLQR_improved_result/' + env_name + '/result.txt', 'w') as f:
        f.write('Average percentage of all models: ' + str(avg_model_percent) + '\n')
        f.write('Best model: ' + best_model + ', best percentage: ' + str(best_model_percent))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run iLQR')
    parser.add_argument('--env', required=True, type=str, help='name of the environment')
    parser.add_argument('--ilqr_iters', required=True, type=int, help='number of ilqr iterations')
    parser.add_argument('--num_uniform', default=5, type=int, help='number of uniform actions trajectories')
    parser.add_argument('--num_extreme', default=5, type=int, help='number of extreme actions trajectories')
    parser.add_argument('--init_mu', default=0.1, type=float, help='the initial value of the invserse regulator')
    parser.add_argument('--mu_mul', default=2, type=float, help='the inverse regulator multiplier')
    parser.add_argument('--mu_max', default=1.0, type=float, help='the maximum inverse regulator')
    parser.add_argument('--init_alpha', default=1.0, type=float, help='the initial alpha in line search')
    parser.add_argument('--alpha_mul', default=0.8, type=float, help='the alpha multiplier')
    parser.add_argument('--alpha_min', default=0.5, type=float, help='the minimum inverse regulator')

    args = parser.parse_args()

    main(args)
