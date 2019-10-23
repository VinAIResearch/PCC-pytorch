from tensorboardX import SummaryWriter
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
import sys

from pcc_model import PCC
from datasets import *
from losses import *
from data.planar.sample_planar import sample as planar_sampler
from data.planar.sample_planar import PlanarEnv
from gym.envs.classic_control import PendulumEnv
from data.pendulum.sample_pendulum import sample as pendulum_sampler

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

device = torch.device("cuda")
envs = {'planar': PlanarEnv, 'pendulum': PendulumEnv}
samplers = {'planar': planar_sampler, 'pendulum': pendulum_sampler}
datasets = {'planar': PlanarDataset, 'pendulum': PendulumDataset}
settings = {'planar': (1600, 2, 2), 'pendulum': (4608, 3, 1)}
num_eval = 10 # number of images evaluated on tensorboard

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

def train(model, train_loader, lam, vae_coeff, determ_coeff, optimizer, armotized):
    avg_pred_loss = 0.0
    avg_consis_loss = 0.0
    avg_cur_loss = 0.0
    avg_loss = 0.0  
    
    num_batches = len(train_loader)
    model.train()
    for x, u, x_next in train_loader:
        x = x.to(device).double().view(-1, model.x_dim)
        u = u.to(device).double()
        x_next = x_next.to(device).double().view(-1, model.x_dim)
        # x = x.view(-1, model.x_dim)
        # x_next = x_next.view(-1, model.x_dim)
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

def compute_log_likelihood(x, x_recon, x_next, x_next_pred):
    loss_1 = -torch.mean(torch.sum(x * torch.log(1e-8 + x_recon)
                                   + (1 - x) * torch.log(1e-8 + 1 - x_recon), dim=1))
    loss_2 = -torch.mean(torch.sum(x_next * torch.log(1e-8 + x_next_pred)
                                   + (1 - x_next) * torch.log(1e-8 + 1 - x_next_pred), dim=1))
    return loss_1, loss_2

def evaluate(model, test_loader):
    model.eval()
    num_batches = len(test_loader)
    state_loss, next_state_loss = 0., 0.
    with torch.no_grad():
        for x, u, x_next in test_loader:
            x = x.to(device).double().view(-1, model.x_dim)
            u = u.to(device).double()
            x_next = x_next.to(device).double().view(-1, model.x_dim)

            x_recon, x_next_pred = model.predict(x, u)
            loss_1, loss_2 = compute_log_likelihood(x, x_recon, x_next, x_next_pred)
            state_loss += loss_1
            next_state_loss += loss_2

    return state_loss.item() / num_batches, next_state_loss.item() / num_batches

# code for visualizing the training process
def predict_x_next(model, env_name, num_eval):
    model.eval()
    # frist sample a true trajectory from the environment
    sampler = samplers[env_name]
    env = envs[env_name]()
    state_samples, sampled_data = sampler(env, num_eval)

    # use the trained model to predict the next observation
    predicted = []
    for x, u, x_next in sampled_data:
        x_reshaped = x.reshape(-1)
        x_reshaped = torch.from_numpy(x_reshaped).to(device).double().unsqueeze(dim=0)
        u = torch.from_numpy(u).double().to(device).unsqueeze(dim=0)
        with torch.no_grad():
            _, x_next_pred = model.predict(x_reshaped, u)
        predicted.append(x_next_pred.squeeze().cpu().numpy().reshape(env.width, env.height))
    true_x_next = [data[-1] for data in sampled_data]
    return true_x_next, predicted

def plot_preds(model, env, num_eval):
    true_x_next, pred_x_next = predict_x_next(model, env, num_eval)

    # plot the predicted and true observations
    fig, axes =plt.subplots(nrows=2, ncols=num_eval)
    plt.setp(axes, xticks=[], yticks=[])
    pad = 5
    axes[0, 0].annotate('True observations', xy=(0, 0.5), xytext=(-axes[0,0].yaxis.labelpad - pad, 0),
                   xycoords=axes[0,0].yaxis.label, textcoords='offset points',
                   size='large', ha='right', va='center')
    axes[1, 0].annotate('Predicted observations', xy=(0, 0.5), xytext=(-axes[1, 0].yaxis.labelpad - pad, 0),
                        xycoords=axes[1, 0].yaxis.label, textcoords='offset points',
                        size='large', ha='right', va='center')

    for idx in np.arange(num_eval):
        axes[0, idx].imshow(true_x_next[idx], cmap='Greys')
        axes[1, idx].imshow(pred_x_next[idx], cmap='Greys')
    fig.tight_layout()
    return fig

def main(args):
    env_name = args.env
    armotized = args.armotized
    assert env_name in ['planar', 'pendulum']
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
    train_set = dataset('data/' + env_name, train=True)
    test_set = dataset('data/' + env_name, train=False)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)

    x_dim, z_dim, u_dim = settings[env_name]
    model = PCC(armotized = armotized, x_dim=x_dim, z_dim=z_dim, u_dim=u_dim, env=env_name).to(device)

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

    for i in range(epoches):
        avg_pred_loss, avg_consis_loss, avg_cur_loss, avg_loss = train(model, train_loader, lam, vae_coeff, determ_coeff, optimizer, armotized)
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

        # save model
        if (i + 1) % iter_save == 0:
            # writer.add_figure('actual vs. predicted observations',
            #                   plot_preds(model, env_name, num_eval),
            #                   global_step=i)
            print('Saving the model.............')

            torch.save(model.state_dict(), result_path + '/model_' + str(i + 1))
            with open(result_path + '/loss_' + str(i + 1), 'w') as f:
                f.write('\n'.join(['Prediction loss: ' + str(avg_pred_loss), 
                                'Consistency loss: ' + str(avg_consis_loss), 
                                'Curvature loss: ' + str(avg_cur_loss),
                                'Training loss: ' + str(avg_loss)
                                ]))

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