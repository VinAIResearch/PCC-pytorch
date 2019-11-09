import argparse
from torchvision.transforms import ToTensor
import torch

from pcc_model import PCC
from mdp.plane_obstacles_mdp import PlanarObstaclesMDP
from mdp.pole_simple_mdp import VisualPoleSimpleSwingUp
from mdp.cartpole_mdp import VisualCartPoleBalance
from ilqr_improved_utils import *
from datasets import *

np.random.seed(0)
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

config_path = {'plane': 'ilqr_config/plane.json', 'swing': 'ilqr_config/swing_up.json', 'balance': 'ilqr_config/balance.json', 'cartpole': 'ilqr_config/cartpole.json'}
envs = {'plane': 'planar', 'swing': 'pendulum', 'balance': 'pendulum', 'cartpole': 'cartpole'}

def main(args):
    task = args.task
    assert task in ['plane', 'swing', 'balance', 'cartpole']
    with open(config_path[task]) as f:
        config = json.load(f)

    # environment specification
    env_name = envs[task]
    horizon = config['horizon_prob']
    plan_len = config['plan_len']
    x_dim = config['obs_shape']
    if task in ['plane', 'swing', 'balance']: # mlp
        x_dim = np.prod(x_dim)
    u_dim = config['action_dim']
    z_dim = config['latent_dim']

    # ilqr specification
    R_z = config['q_weight'] * np.eye(z_dim)
    R_u = config['r_weight'] * np.eye(u_dim)
    num_uniform = config['uniform_trajs']
    num_extreme = config['extreme_trajs']
    ilqr_iters = config['ilqr_iters']
    inv_regulator_init = config['pinv_init']
    inv_regulator_multi = config['pinv_mult']
    inv_regulator_max = config['pinv_max']
    alpha_init = config['alpha_init']
    alpha_mult = config['alpha_mult']
    alpha_min = config['alpha_min']

    # the epoch number the model was saved at
    epoch = config['epoch']

    # the folder where all trained models are saved
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
        model_path = 'iLQR_improved_result/' + task + '/' + log_base
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        print('iLQR for ' + log_base)

        # load the trained model
        model = PCC(armotized, x_dim, z_dim, u_dim, env_name)
        model.load_state_dict(torch.load(log + '/model_' + str(epoch)))
        model.eval()
        dynamics = model.dynamics
        encoder = model.encoder

        # run iLQR for a particular model
        avg_percent = 0.0
        for random_task_id in range(10):  # perform 10 random tasks (10 different start states and goal states)
            print('Performing task ' + str(random_task_id + 1))
            # sample random start and goal state
            s_start_min, s_start_max = config['start_min'], config['start_max']
            s_start = np.random.uniform(low=s_start_min, high=s_start_max)
            s_goal = config['goal'][np.random.choice(len(config['goal']))]
            s_goal = np.array(s_goal)

            # mdp
            if task == 'plane':
                mdp = PlanarObstaclesMDP(goal=s_goal, goal_thres=config['distance_thresh'],
                                         noise=config['noise'])
            if task in ['swing', 'balance']:
                mdp = VisualPoleSimpleSwingUp(frequency=config['frequency'],
                                              noise=config['noise'], torque=config['torque'])
            if task == 'cartpole':
                mdp = VisualCartPoleBalance(frequency=config['frequency'], noise=config['noise'])
            # get z_start and z_goal
            image_start, x_start = get_x_data(mdp, s_start, config)
            image_goal, x_goal = get_x_data(mdp, s_goal, config)
            with torch.no_grad():
                z_start, _ = encoder(x_start)
                z_goal, _ = encoder(x_goal)
            z_start = z_start.squeeze().numpy()
            z_goal = z_goal.squeeze().numpy()

            # initialize actions trajectories
            all_actions_trajs = random_actions_trajs(mdp, num_uniform, num_extreme, plan_len)
            actions_final = []
            # perform reciding horizon iLQR
            s_start_horizon = np.copy(s_start)  # start state will be changed at each horizon
            image_start_horizon = np.copy(image_start)
            z_start_horizon = np.copy(z_start)
            for plan_iter in range(1, horizon + 1):
                print('Planning for horizon ' + str(plan_iter))
                latent_cost_list = [None] * len(all_actions_trajs)
                # iterate over all trajectories
                for traj_id in range(len(all_actions_trajs)):
                    print('Running iLQR for trajectory ' + str(traj_id + 1))
                    # initialize the inverse regulator
                    inv_regulator = inv_regulator_init
                    for iter in range(1, ilqr_iters + 1):
                        # compute the latent trajectory
                        z_seq = compute_latent_traj(z_start_horizon,
                                                    all_actions_trajs[traj_id], dynamics)
                        # compute the linearization matrices
                        A_seq, B_seq = seq_jacobian(dynamics, z_seq, all_actions_trajs[traj_id])
                        # run backward
                        k_small, K_big = backward(R_z, R_u, z_seq, all_actions_trajs[traj_id],
                                                  z_goal, A_seq, B_seq, inv_regulator)
                        # compute the latent cost for current u_seq
                        current_cost = latent_cost(R_z, R_u, z_seq, z_goal, all_actions_trajs[traj_id])
                        latent_cost_list[traj_id] = current_cost
                        # forward using line search
                        alpha = alpha_init
                        accept = False  # if any alpha is accepted
                        while alpha > alpha_min:
                            u_seq_cand = forward(all_actions_trajs[traj_id], k_small, K_big, A_seq, B_seq, alpha)
                            z_seq_cand = compute_latent_traj(z_start_horizon,
                                                             u_seq_cand, dynamics)
                            # z_seq_cand, u_seq_cand = forward(z_seq, all_actions_trajs[traj_id], k_small, K_big, dynamics, alpha)
                            cost_cand = latent_cost(R_z, R_u, z_seq_cand, z_goal, u_seq_cand)
                            if cost_cand < current_cost:  # accept the trajectory candidate
                                accept = True
                                all_actions_trajs[traj_id] = u_seq_cand
                                latent_cost_list[traj_id] = cost_cand
                                print('Found a better action sequence. New latent cost: ' + str(cost_cand.item()))
                                break
                            else:
                                alpha *= alpha_mult
                        if accept:
                            inv_regulator = inv_regulator_init
                        else:
                            inv_regulator *= inv_regulator_multi
                        if inv_regulator > inv_regulator_max:
                            print('Can not improve. Stop iLQR.')
                            break

                    print('===================================================================')

                traj_opt_id = np.argmin(latent_cost_list)
                action_chosen = all_actions_trajs[traj_opt_id][0]
                actions_final.append(action_chosen)
                all_actions_trajs = refresh_actions_trajs(all_actions_trajs, traj_opt_id, mdp,
                                                          np.min([plan_len, horizon - plan_iter]),
                                                          num_uniform, num_extreme)
                if env_name == 'planar':
                    s_start_horizon = mdp.transition_function(s_start_horizon, action_chosen)
                    image_start_horizon, x_start_new = get_x_data(mdp, s_start_horizon, config)
                elif env_name == 'pendulum':
                    s_start_horizon, image_start_next_horizon = mdp.transition_function((s_start_horizon, image_start_horizon), action_chosen)
                    image_start_next_horizon = image_start_next_horizon.squeeze()
                    image_start_stacked = np.vstack((image_start_horizon, image_start_next_horizon))
                    x_start_horizon = torch.from_numpy(image_start_stacked).view(x_dim).unsqueeze(0).double()
                    with torch.no_grad():
                        z_start_horizon, _ = model.encode(x_start_horizon)
                    z_start_horizon = z_start_horizon.squeeze().numpy()
                    image_start_horizon = image_start_next_horizon
                elif env_name == 'cartpole':
                    s_start_horizon, image_start_next_horizon = mdp.transition_function(
                        (s_start_horizon, image_start_horizon), action_chosen)
                    image_start_next_horizon = image_start_next_horizon.squeeze()
                    x_start_horizon = torch.zeros(size=(2, 80, 80))
                    x_start_horizon[0, :, :] = torch.from_numpy(image_start_horizon)
                    x_start_horizon[1, :, :] = torch.from_numpy(image_start_next_horizon)
                    x_start_horizon = x_start_horizon.unsqueeze(0)
                    with torch.no_grad():
                        z_start_horizon, _ = model.encode(x_start_horizon)
                    z_start_horizon = z_start_horizon.squeeze().numpy()
                    image_start_horizon = image_start_next_horizon

            obs_traj, goal_counter = traj_opt_actions(s_start, image_start, actions_final, mdp, env_name)
            # compute the percentage close to goal
            percent = goal_counter / horizon
            avg_percent += percent
            with open(model_path + '/result.txt', 'a+') as f:
                f.write('Task {:01d}: '.format(random_task_id + 1) + str(percent) + '\n')

            # save trajectory as gif file
            gif_path = model_path + '/task_{:01d}.gif'.format(random_task_id + 1)
            save_traj(obs_traj, image_goal, gif_path, task)

        avg_percent = avg_percent / 10
        avg_model_percent += avg_percent
        if avg_percent > best_model_percent:
            best_model = log_base
            best_model_percent = avg_percent
        with open(model_path + '/result.txt', 'a+') as f:
            f.write('Average percentage: ' + str(avg_percent))

    avg_model_percent = avg_model_percent / len(log_folders)
    with open('iLQR_improved_result/' + task + '/result.txt', 'w') as f:
        f.write('Average percentage of all models: ' + str(avg_model_percent) + '\n')
        f.write('Best model: ' + best_model + ', best percentage: ' + str(best_model_percent))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run iLQR')
    parser.add_argument('--task', required=True, type=str, help='name of the environment')
    args = parser.parse_args()

    main(args)
