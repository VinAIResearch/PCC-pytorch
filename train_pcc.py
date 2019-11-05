from tensorboardX import SummaryWriter
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import argparse

from pcc_model import PCC
from datasets import *
from losses import *

from mdp.plane_obstacles_mdp import PlanarObstaclesMDP
from mdp.pole_simple_mdp import VisualPoleSimpleSwingUp
from mdp.cartpole_mdp import VisualCartPoleBalance
from latent_map_evolve import *

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

device = torch.device("cuda")
mdps = {'planar': PlanarObstaclesMDP, 'pendulum': VisualPoleSimpleSwingUp, 'cartpole': VisualCartPoleBalance}
datasets = {'planar': PlanarDataset, 'pendulum': PendulumDataset, 'cartpole': CartPoleDataset}
dims = {'planar': (1600, 2, 2), 'pendulum': (4608, 3, 1), 'cartpole': ((2, 80, 80), 8, 1)}

def compute_loss(model, armotized, x, x_next,
                x_next_recon,
                mu_q_z, logvar_q_z, mu_p_z, logvar_p_z,
                mu_q_z_next, logvar_q_z_next,
                z_next, mu_p_z_next, logvar_p_z_next,
                z, u, x_recon, x_next_determ,
                lam=(1.0,8.0,8.0), delta=0.1, vae_coeff=0.01, determ_coeff=0.3):
    # prediction loss
    pred_loss  = - bernoulli(x_next, x_next_recon) \
                + KL(mu_q_z, logvar_q_z, mu_p_z, logvar_p_z) \
                - entropy(mu_q_z_next, logvar_q_z_next) \
                - gaussian(z_next, mu_p_z_next, logvar_p_z_next)

    # consistency loss
    consis_loss = - entropy(mu_q_z_next, logvar_q_z_next) \
                - gaussian(z_next, mu_p_z_next, logvar_p_z_next) \
                + KL(mu_q_z, logvar_q_z, mu_p_z, logvar_p_z) \

    # curvature loss
    cur_loss = curvature(model, z, u, delta, armotized)
    # cur_loss = curvature_variant(model, z, u, delta, armotized)

    # additional vae loss
    vae_loss = vae_bound(x, x_recon, mu_p_z, logvar_p_z)

    # additional deterministic loss
    determ_loss = -bernoulli(x_next, x_next_determ)
    
    lam_p, lam_c, lam_cur = lam
    return pred_loss, consis_loss, cur_loss, \
            lam_p * pred_loss + lam_c * consis_loss + lam_cur * cur_loss + vae_coeff * vae_loss + determ_coeff * determ_loss

def train(env_name, model, train_loader, lam, vae_coeff, determ_coeff, optimizer, armotized):
    avg_pred_loss = 0.0
    avg_consis_loss = 0.0
    avg_cur_loss = 0.0
    avg_loss = 0.0
    
    num_batches = len(train_loader)
    model.train()
    for x, u, x_next in train_loader:
        if env_name == 'planar' or env_name == 'pendulum':
            x = x.view(-1, model.x_dim)
            x_next = x_next.view(-1, model.x_dim)
        x = x.to(device).double()
        u = u.to(device).double()
        x_next = x_next.to(device).double()
        optimizer.zero_grad()

        x_next_recon, \
        mu_q_z, logvar_q_z, mu_p_z, logvar_p_z, \
        mu_q_z_next, logvar_q_z_next, \
        z_next, mu_p_z_next, logvar_p_z_next, \
        z, u, x_recon, x_next_determ = model(x, u, x_next)

        pred_loss, consis_loss, cur_loss, loss = compute_loss(
                model, armotized, x, x_next,
                x_next_recon,
                mu_q_z, logvar_q_z, mu_p_z, logvar_p_z,
                mu_q_z_next, logvar_q_z_next,
                z_next, mu_p_z_next, logvar_p_z_next,
                z, u, x_recon, x_next_determ,
                lam=lam, vae_coeff=vae_coeff, determ_coeff=determ_coeff
                )

        avg_pred_loss += pred_loss.item()
        avg_consis_loss += consis_loss.item()
        avg_cur_loss += cur_loss.item()
        avg_loss += loss.item()
        loss.backward()
        optimizer.step()

    return avg_pred_loss / num_batches, avg_consis_loss / num_batches, avg_cur_loss / num_batches, avg_loss / num_batches

def main(args):
    env_name = args.env
    assert env_name in ['planar', 'pendulum', 'cartpole']
    armotized = args.armotized
    log_dir = args.log_dir
    seed = args.seed
    batch_size = args.batch_size
    lam_p = args.lam_p
    lam_c = args.lam_c
    lam_cur = args.lam_cur
    lam = (lam_p, lam_c, lam_cur)
    vae_coeff = args.vae_coeff
    determ_coeff = args.determ_coeff
    lr = args.lr
    weight_decay = args.decay
    epoches = args.num_iter
    iter_save = args.iter_save

    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = datasets[env_name]
    data = dataset('data/' + env_name)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)

    x_dim, z_dim, u_dim = dims[env_name]
    model = PCC(armotized=armotized, x_dim=x_dim, z_dim=z_dim, u_dim=u_dim, env=env_name).to(device)

    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999), eps=1e-8, lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=int(epoches / 3), gamma=0.5)

    log_path = 'logs/' + env_name + '/' + log_dir
    if not path.exists(log_path):
        os.makedirs(log_path)
    writer = SummaryWriter(log_path)

    result_path = 'result/' + env_name + '/' + log_dir
    if not path.exists(result_path):
        os.makedirs(result_path)
    with open(result_path + '/settings', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # latent_maps = [draw_latent_map(model, mdp)]
    for i in range(epoches):
        avg_pred_loss, avg_consis_loss, avg_cur_loss, avg_loss = train(env_name, model, data_loader, lam, vae_coeff, determ_coeff, optimizer, armotized)
        scheduler.step()
        print('Epoch %d' % i)
        print("Prediction loss: %f" % (avg_pred_loss))
        print("Consistency loss: %f" % (avg_consis_loss))
        print("Curvature loss: %f" % (avg_cur_loss))
        print("Training loss: %f" % (avg_loss))
        print ('--------------------------------------')

        # ...log the running loss
        writer.add_scalar('prediction loss', avg_pred_loss, i)
        writer.add_scalar('consistency loss', avg_consis_loss, i)
        writer.add_scalar('curvature loss', avg_cur_loss, i)
        writer.add_scalar('training loss', avg_loss, i)
        # if (i+1) % 10 == 0:
        #     map_i = draw_latent_map(model, mdp)
        #     latent_maps.append(map_i)
        # save model
        if (i + 1) % iter_save == 0:
            print('Saving the model.............')

            torch.save(model.state_dict(), result_path + '/model_' + str(i + 1))
            with open(result_path + '/loss_' + str(i + 1), 'w') as f:
                f.write('\n'.join(['Prediction loss: ' + str(avg_pred_loss), 
                                'Consistency loss: ' + str(avg_consis_loss), 
                                'Curvature loss: ' + str(avg_cur_loss),
                                'Training loss: ' + str(avg_loss)
                                ]))
    # latent_maps[0].save(result_path + '/latent_map.gif', format='GIF', append_images=latent_maps[1:], save_all=True, duration=100, loop=0)
    writer.close()

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train pcc model')

    # the default value is used for the planar task
    parser.add_argument('--env', required=True, type=str, help='environment used for training')
    parser.add_argument('--armotized', required=True, type=str2bool, nargs='?',
                        const=True, default=False, help='type of dynamics model')
    parser.add_argument('--log_dir', required=True, type=str, help='directory to save training log')
    parser.add_argument('--seed', required=True, type=int, help='seed number')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--lam_p', default=1.0, type=float, help='weight of prediction loss')
    parser.add_argument('--lam_c', default=8.0, type=float, help='weight of consistency loss')
    parser.add_argument('--lam_cur', default=8.0, type=float, help='weight of curvature loss')
    parser.add_argument('--vae_coeff', default=0.01, type=float, help='coefficient of additional vae loss')
    parser.add_argument('--determ_coeff', default=0.3, type=float, help='coefficient of addtional deterministic loss')
    parser.add_argument('--lr', default=0.0005, type=float, help='learning rate')
    parser.add_argument('--decay', default=0.001, type=float, help='L2 regularization')
    parser.add_argument('--num_iter', default=5000, type=int, help='number of epoches')
    parser.add_argument('--iter_save', default=1000, type=int, help='save model and result after this number of iterations')

    args = parser.parse_args()

    main(args)
