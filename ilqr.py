import argparse
import json
import os
import random

import numpy as np
import torch
from ilqr_utils import (
    backward,
    compute_latent_traj,
    forward,
    get_x_data,
    latent_cost,
    random_actions_trajs,
    refresh_actions_trajs,
    save_traj,
    seq_jacobian,
    update_horizon_start,
)
from mdp.cartpole_mdp import CartPoleMDP
from mdp.pendulum_mdp import PendulumMDP
from mdp.plane_obstacles_mdp import PlanarObstaclesMDP
from mdp.three_pole_mdp import ThreePoleMDP
from pcc_model import PCC


seed = 2020
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.set_default_dtype(torch.float64)

config_path = {
    "plane": "ilqr_config/plane.json",
    "swing": "ilqr_config/swing.json",
    "balance": "ilqr_config/balance.json",
    "cartpole": "ilqr_config/cartpole.json",
    "threepole": "ilqr_config/threepole.json",
}
env_data_dim = {
    "planar": (1600, 2, 2),
    "pendulum": ((2, 48, 48), 3, 1),
    "cartpole": ((2, 80, 80), 8, 1),
    "threepole": ((2, 80, 80), 8, 3),
}


def main(args):
    task_name = args.task
    assert task_name in ["planar", "balance", "swing", "cartpole", "threepole", "pendulum_gym", "mountain_car"]
    env_name = "pendulum" if task_name in ["balance", "swing"] else task_name

    setting_path = args.setting_path
    setting = os.path.basename(os.path.normpath(setting_path))
    noise = args.noise
    epoch = args.epoch
    x_dim, z_dim, u_dim = env_data_dim[env_name]
    if env_name in ["planar", "pendulum"]:
        x_dim = np.prod(x_dim)

    ilqr_result_path = "iLQR_result/" + "_".join([task_name, str(setting), str(noise), str(epoch)])
    if not os.path.exists(ilqr_result_path):
        os.makedirs(ilqr_result_path)
    with open(ilqr_result_path + "/settings", "w") as f:
        json.dump(args.__dict__, f, indent=2)

    # each trained model will perform 10 random tasks
    all_task_configs = []
    for task_counter in range(10):
        # config for this task
        with open(config_path[task_name]) as f:
            config = json.load(f)

        # sample random start and goal state
        s_start_min, s_start_max = config["start_min"], config["start_max"]
        config["s_start"] = np.random.uniform(low=s_start_min, high=s_start_max)
        s_goal = config["goal"][np.random.choice(len(config["goal"]))]
        config["s_goal"] = np.array(s_goal)

        all_task_configs.append(config)

    # the folder where all trained models are saved
    log_folders = [
        os.path.join(setting_path, dI)
        for dI in os.listdir(setting_path)
        if os.path.isdir(os.path.join(setting_path, dI))
    ]
    log_folders.sort()

    # statistics on all trained models
    avg_model_percent = 0.0
    best_model_percent = 0.0
    for log in log_folders:
        with open(log + "/settings", "r") as f:
            settings = json.load(f)
            armotized = settings["armotized"]

        log_base = os.path.basename(os.path.normpath(log))
        model_path = ilqr_result_path + "/" + log_base
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        print("iLQR for " + log_base)

        # load the trained model
        model = PCC(armotized, x_dim, z_dim, u_dim, env_name)
        model.load_state_dict(torch.load(log + "/model_" + str(epoch), map_location="cpu"))
        model.eval()
        dynamics = model.dynamics
        encoder = model.encoder

        # run the task with 10 different start and goal states for a particular model
        avg_percent = 0.0
        for task_counter, config in enumerate(all_task_configs):

            print("Performing task %d: " % (task_counter) + str(config["task"]))

            # environment specification
            horizon = config["horizon_prob"]
            plan_len = config["plan_len"]

            # ilqr specification
            R_z = config["q_weight"] * np.eye(z_dim)
            R_u = config["r_weight"] * np.eye(u_dim)
            num_uniform = config["uniform_trajs"]
            num_extreme = config["extreme_trajs"]
            ilqr_iters = config["ilqr_iters"]
            inv_regulator_init = config["pinv_init"]
            inv_regulator_multi = config["pinv_mult"]
            inv_regulator_max = config["pinv_max"]
            alpha_init = config["alpha_init"]
            alpha_mult = config["alpha_mult"]
            alpha_min = config["alpha_min"]

            s_start = config["s_start"]
            s_goal = config["s_goal"]

            # mdp
            if env_name == "planar":
                mdp = PlanarObstaclesMDP(goal=s_goal, goal_thres=config["distance_thresh"], noise=noise)
            elif env_name == "pendulum":
                mdp = PendulumMDP(frequency=config["frequency"], noise=noise, torque=config["torque"])
            elif env_name == "cartpole":
                mdp = CartPoleMDP(frequency=config["frequency"], noise=noise)
            elif env_name == "threepole":
                mdp = ThreePoleMDP(frequency=config["frequency"], noise=noise, torque=config["torque"])
            # get z_start and z_goal
            x_start = get_x_data(mdp, s_start, config)
            x_goal = get_x_data(mdp, s_goal, config)
            with torch.no_grad():
                z_start = encoder(x_start).mean
                z_goal = encoder(x_goal).mean
            z_start = z_start.squeeze().numpy()
            z_goal = z_goal.squeeze().numpy()

            # initialize actions trajectories
            all_actions_trajs = random_actions_trajs(mdp, num_uniform, num_extreme, plan_len)

            # perform reciding horizon iLQR
            s_start_horizon = np.copy(s_start)  # s_start and z_start is changed at each horizon
            z_start_horizon = np.copy(z_start)
            obs_traj = [mdp.render(s_start).squeeze()]
            goal_counter = 0.0
            for plan_iter in range(1, horizon + 1):
                latent_cost_list = [None] * len(all_actions_trajs)
                # iterate over all trajectories
                for traj_id in range(len(all_actions_trajs)):
                    # initialize the inverse regulator
                    inv_regulator = inv_regulator_init
                    for iter in range(1, ilqr_iters + 1):
                        u_seq = all_actions_trajs[traj_id]
                        z_seq = compute_latent_traj(z_start_horizon, u_seq, dynamics)
                        # compute the linearization matrices
                        A_seq, B_seq = seq_jacobian(dynamics, z_seq, u_seq)
                        # run backward
                        k_small, K_big = backward(R_z, R_u, z_seq, u_seq, z_goal, A_seq, B_seq, inv_regulator)
                        current_cost = latent_cost(R_z, R_u, z_seq, z_goal, u_seq)
                        # forward using line search
                        alpha = alpha_init
                        accept = False  # if any alpha is accepted
                        while alpha > alpha_min:
                            z_seq_cand, u_seq_cand = forward(
                                z_seq, all_actions_trajs[traj_id], k_small, K_big, dynamics, alpha
                            )
                            cost_cand = latent_cost(R_z, R_u, z_seq_cand, z_goal, u_seq_cand)
                            if cost_cand < current_cost:  # accept the trajectory candidate
                                accept = True
                                all_actions_trajs[traj_id] = u_seq_cand
                                latent_cost_list[traj_id] = cost_cand
                                break
                            else:
                                alpha *= alpha_mult
                        if accept:
                            inv_regulator = inv_regulator_init
                        else:
                            inv_regulator *= inv_regulator_multi
                        if inv_regulator > inv_regulator_max:
                            break

                for i in range(len(latent_cost_list)):
                    if latent_cost_list[i] is None:
                        latent_cost_list[i] = np.inf
                traj_opt_id = np.argmin(latent_cost_list)
                action_chosen = all_actions_trajs[traj_opt_id][0]
                s_start_horizon, z_start_horizon = update_horizon_start(
                    mdp, s_start_horizon, action_chosen, encoder, config
                )

                obs_traj.append(mdp.render(s_start_horizon).squeeze())
                goal_counter += mdp.reward_function(s_start_horizon)

                all_actions_trajs = refresh_actions_trajs(
                    all_actions_trajs,
                    traj_opt_id,
                    mdp,
                    np.min([plan_len, horizon - plan_iter]),
                    num_uniform,
                    num_extreme,
                )

            # compute the percentage close to goal
            success_rate = goal_counter / horizon
            print("Success rate: %.2f" % (success_rate))
            percent = success_rate
            avg_percent += success_rate
            with open(model_path + "/result.txt", "a+") as f:
                f.write(config["task"] + ": " + str(percent) + "\n")

            # save trajectory as gif file
            gif_path = model_path + "/task_{:01d}.gif".format(task_counter + 1)
            save_traj(obs_traj, mdp.render(s_goal).squeeze(), gif_path, config["task"])

        avg_percent = avg_percent / 10
        print("Average success rate: " + str(avg_percent))
        print("====================================")
        avg_model_percent += avg_percent
        if avg_percent > best_model_percent:
            best_model = log_base
            best_model_percent = avg_percent
        with open(model_path + "/result.txt", "a+") as f:
            f.write("Average percentage: " + str(avg_percent))

    avg_model_percent = avg_model_percent / len(log_folders)
    with open(ilqr_result_path + "/result.txt", "w") as f:
        f.write("Average percentage of all models: " + str(avg_model_percent) + "\n")
        f.write("Best model: " + best_model + ", best percentage: " + str(best_model_percent))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run iLQR")
    parser.add_argument("--task", required=True, type=str, help="task to perform")
    parser.add_argument("--setting_path", required=True, type=str, help="path to load trained models")
    parser.add_argument("--noise", type=float, default=0.0, help="noise level for mdp")
    parser.add_argument("--epoch", type=int, default=2000, help="number of epochs to load model")
    args = parser.parse_args()

    main(args)
