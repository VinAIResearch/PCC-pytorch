import torch
import math

torch.set_default_dtype(torch.float64)

def bernoulli(x, p, iwae=False):
    # if using -> dont take mean
    log_x_p = torch.sum(x * torch.log(1e-8 + p)
                                + (1 - x) * torch.log(1e-8 + 1 - p), dim=-1)
    if not iwae:
        log_x_p = torch.mean(log_x_p)
    return log_x_p

def KL(mu1, logvar1, mu2, logvar2, iwae=False):
    var1 = torch.exp(logvar1)
    var2 = torch.exp(logvar2)
    d = mu1.size(-1)
    kl = 0.5 * torch.sum(
        logvar2 - logvar1 - 1
        + var1 / var2
        + (mu2 - mu1)**2 / var2, dim=-1)
    if not iwae:
        kl = torch.mean(kl)
    return kl

def entropy(mu, logvar, iwae = False):
    d = mu.size(-1)
    H = d + d * torch.log(2 * torch.Tensor([math.pi]).cuda()) \
        + torch.sum(logvar, dim=-1)
    if not iwae:
        H = torch.mean(H)
    return 0.5 * H

def gaussian(z, mu, logvar, iwae=False):
    d = mu.size(-1)
    var = torch.exp(logvar)
    sum = lambda x: torch.sum(x, dim=-1)
    log_z_mu_logvar = sum((z - mu)**2 / var) \
        + d * torch.log(2 * torch.Tensor([math.pi]).cuda()) \
        + sum(logvar)
    if not iwae:
        log_z_mu_logvar = torch.mean(log_z_mu_logvar)
    return -0.5 * log_z_mu_logvar

def curvature(model, z, u, delta, armotized):
    z_alias = z.detach().requires_grad_(True)
    u_alias = u.detach().requires_grad_(True)
    eps_z = torch.normal(mean=torch.zeros_like(z), std=torch.empty_like(z).fill_(delta))
    eps_u = torch.normal(mean=torch.zeros_like(u), std=torch.empty_like(u).fill_(delta))
    # eps_z = torch.normal(0.0, delta, size=z.size()).cuda()
    # eps_u = torch.normal(0.0, delta, size=u.size()).cuda()
    z_bar = z_alias + eps_z
    u_bar = u_alias + eps_u
    f_Z = model.dynamics
    f_z_bar, _, A_bar, B_bar = f_Z(z_bar, u_bar)
    f_z, _, A, B = f_Z(z_alias, u_alias)
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

def vae_bound(x, x_recon, mu_z, logvar_z):
    recon_loss = -bernoulli(x, x_recon)
    regularization_loss = KL(mu_z, logvar_z,
                            torch.zeros_like(mu_z), torch.zeros_like(logvar_z))
    return recon_loss + regularization_loss