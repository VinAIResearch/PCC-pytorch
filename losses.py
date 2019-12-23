import torch
from torch.distributions.kl import kl_divergence

from networks import MultivariateNormalDiag

torch.set_default_dtype(torch.float64)

def bernoulli(x, p):
    p = p.probs
    log_p_x = torch.sum(x * torch.log(1e-10 + p)
                                + (1 - x) * torch.log(1e-10 + 1 - p), dim=-1)
    log_p_x = torch.mean(log_p_x)
    return log_p_x

#def bernoulli(x, p):
    #log_p_x = p.log_prob(x)
    #log_p_x = torch.mean(log_p_x)
    #return log_p_x

def KL(normal_1, normal_2):
    kl = kl_divergence(normal_1, normal_2)
    kl = torch.mean(kl)
    return kl

def entropy(p):
    H = p.entropy()
    H = torch.mean(H)
    return H

def gaussian(z, p):
    log_p_z = p.log_prob(z)
    log_p_z = torch.mean(log_p_z)
    return log_p_z

def vae_bound(x, p_x, p_z):
    recon_loss = -bernoulli(x, p_x)
    regularization_loss = KL(p_z, MultivariateNormalDiag(torch.zeros_like(p_z.mean),
                                                         torch.ones_like(p_z.stddev)))
    return recon_loss + regularization_loss

def ae_loss(x, p_x):
    recon_loss = -bernoulli(x, p_x)
    return recon_loss

def curvature(model, z, u, delta, armotized):
    z_alias = z.detach().requires_grad_(True)
    u_alias = u.detach().requires_grad_(True)
    eps_z = torch.normal(mean=torch.zeros_like(z), std=torch.empty_like(z).fill_(delta))
    eps_u = torch.normal(mean=torch.zeros_like(u), std=torch.empty_like(u).fill_(delta))
    # print ('eps u ' + str(eps_u.size()))
    z_bar = z_alias + eps_z
    u_bar = u_alias + eps_u

    f_z_bar, A_bar, B_bar = model.transition(z_bar, u_bar)
    f_z_bar = f_z_bar.mean
    f_z, A, B = model.transition(z_alias, u_alias)
    f_z = f_z.mean
    # print ('f_z ' + str(f_z.size()))
    if not armotized:
        grad_z, grad_u = torch.autograd.grad(f_z, [z_alias, u_alias], grad_outputs=[eps_z, eps_u], retain_graph=True, create_graph=True)
        taylor_error = f_z_bar - (grad_z + grad_u) - f_z
        cur_loss = torch.mean(torch.sum(taylor_error.pow(2), dim = 1))
    else:
        z_dim, u_dim = z.size(1), u.size(1)
        A_bar = A_bar.view(-1, z_dim, z_dim)
        B_bar = B_bar.view(-1, z_dim, u_dim)
        eps_z = eps_z.view(-1, z_dim, 1)
        eps_u = eps_u.view(-1, u_dim, 1)
        taylor_error = f_z_bar - (torch.bmm(A_bar, eps_z).squeeze() + torch.bmm(B_bar, eps_u).squeeze()) - f_z
        cur_loss = torch.mean(torch.sum(taylor_error.pow(2), dim = 1))
    return cur_loss
