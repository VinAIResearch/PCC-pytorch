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

def new_curvature(model, z, u, delta, armotized):
    z_next, _, _, _ = model.dynamics(z, u)
    temp_z = z - z.mean(dim=0)
    temp_z_next = z_next - z_next.mean(dim=0)
    temp_u = u - u.mean(dim=0)

    cov_z_z_next = torch.sum(temp_z * temp_z_next)**2
    var_prod_z_z_next = torch.sum(temp_z ** 2) * torch.sum(temp_z_next ** 2)

    cov_u_z_next = torch.sum(temp_u * temp_z_next)**2
    var_prod_u_z_next = torch.sum(temp_u ** 2) * torch.sum(temp_z_next ** 2)
    # print ('z next u: ' + str(cov_z_z_next / var_prod_z_z_next))
    # return - cov_z_z_next / var_prod_z_z_next - cov_u_z_next / var_prod_u_z_next
    return - cov_z_z_next / var_prod_z_z_next

def vae_bound(x, x_recon, mu_z, logvar_z):
    recon_loss = -bernoulli(x, x_recon)
    regularization_loss = KL(mu_z, logvar_z,
                            torch.zeros_like(mu_z), torch.zeros_like(logvar_z))
    return recon_loss + regularization_loss

def partial_iwae_loss(model, x, u, x_next, x_next_recon, mu_q_z_next, logvar_q_z_next, z_next,
                      mu_p_z, logvar_p_z, mu_q_z, logvar_q_z, k):
    """
    :param mu_q_z_next: q(z_t+1 | x_t+1)
    :param logvar_q_z_next: q(z_t+1 | x_t+1)
    :param mu_q_z: q(z_t | z_t+1, x_t, u_t)
    :param logvar_q_z: q(z_t | z_t+1, x_t, u_t)
    :param k: number of particles
    :return:
    """
    u_rep = u.expand((k, u.size(0), u.size(1)))
    x_next_rep = x_next.expand((k, x_next.size(0), x_next.size(1)))
    z_next_rep = z_next.repeat(k, 1, 1)

    # sample k particles
    mu_q_z_rep = mu_q_z.repeat(k, 1, 1)
    logvar_q_z_rep = logvar_q_z.repeat(k, 1, 1)
    z_samples = model.reparam(mu_q_z_rep, logvar_q_z_rep)

    # compute w_i
    mu_p_z_rep, logvar_p_z_rep = mu_p_z.repeat(k, 1, 1), logvar_p_z.repeat(k, 1, 1)
    log_p_z_x = gaussian(z_samples, mu_p_z_rep, logvar_p_z_rep, iwae=True)

    mu_z_next_pred, logvar_z_next_pred, _, _ = model.transition(z_samples, u_rep)
    log_p_z_next_z = gaussian(z_next_rep, mu_z_next_pred, logvar_z_next_pred, iwae=True)

    x_next_recon_rep = x_next_recon.repeat(k, 1, 1)
    log_x_next_z_next = bernoulli(x_next_rep, x_next_recon_rep, iwae=True)

    log_q_z_particle = gaussian(z_samples, mu_q_z_rep, logvar_q_z_rep, iwae=True)

    entropy_q_z_next = entropy(mu_q_z_next, logvar_q_z_next)
    #  normalized w_i * log (w_i)
    # for predition loss
    log_weight_pred = log_p_z_x + log_p_z_next_z + log_x_next_z_next - log_q_z_particle
    log_weight_pred = log_weight_pred - torch.max(log_weight_pred, dim=0)[0]
    with torch.no_grad():
        w_pred = torch.exp(log_weight_pred)
        w_pred = w_pred / torch.sum(w_pred, dim=0)
    pred_loss = -torch.mean(torch.sum(w_pred * (log_p_z_x + log_p_z_next_z + log_x_next_z_next - log_q_z_particle), dim=0)) - entropy_q_z_next

    # for consistency loss
    log_weight_consis = log_p_z_x + log_p_z_next_z - log_q_z_particle
    log_weight_consis = log_weight_consis - torch.max(log_weight_consis, dim=0)[0]
    with torch.no_grad():
        w_consis = torch.exp(log_weight_consis)
        w_consis = w_consis / torch.sum(w_consis, dim=0)
    consis_loss = -torch.mean(torch.sum(w_consis * (log_p_z_x + log_p_z_next_z - log_q_z_particle), dim=0)) - entropy_q_z_next

    return pred_loss, consis_loss

# partial iwae loss for testing
def partial_iwae_test(model, x, u, x_next, x_next_recon, mu_q_z_next, logvar_q_z_next, z_next,
                      mu_p_z, logvar_p_z, mu_q_z, logvar_q_z, k):
    with torch.no_grad():
        u_rep = u.expand((k, u.size(0), u.size(1)))
        x_next_rep = x_next.expand((k, x_next.size(0), x_next.size(1)))
        z_next_rep = z_next.repeat(k, 1, 1)

        # sample k particles
        mu_q_z_rep = mu_q_z.repeat(k, 1, 1)
        logvar_q_z_rep = logvar_q_z.repeat(k, 1, 1)
        z_samples = model.reparam(mu_q_z_rep, logvar_q_z_rep)

        # compute w_i
        mu_p_z_rep, logvar_p_z_rep = mu_p_z.repeat(k, 1, 1), logvar_p_z.repeat(k, 1, 1)
        log_p_z_x = gaussian(z_samples, mu_p_z_rep, logvar_p_z_rep, iwae=True)

        mu_z_next_pred, logvar_z_next_pred, _, _ = model.transition(z_samples, u_rep)
        log_p_z_next_z = gaussian(z_next_rep, mu_z_next_pred, logvar_z_next_pred, iwae=True)

        x_next_recon_rep = x_next_recon.repeat(k, 1, 1)
        log_x_next_z_next = bernoulli(x_next_rep, x_next_recon_rep, iwae=True)

        log_q_z_particle = gaussian(z_samples, mu_q_z_rep, logvar_q_z_rep, iwae=True)

        entropy_q_z_next = entropy(mu_q_z_next, logvar_q_z_next)

        log_weight_pred_test = log_p_z_x + log_p_z_next_z + log_x_next_z_next - log_q_z_particle
        weight_pred_test = torch.exp(log_weight_pred_test)
        pred_loss_test = -torch.mean(torch.log(torch.mean(weight_pred_test, 0))) - entropy_q_z_next

        log_weight_consis_test = log_p_z_x + log_p_z_next_z - log_q_z_particle
        weight_consis_test = torch.exp(log_weight_consis_test)
        consis_loss_test = -torch.mean(torch.log(torch.mean(weight_consis_test, 0))) - entropy_q_z_next

    return pred_loss_test, consis_loss_test

def elbo_test(x_next, x_next_recon, mu_q_z, logvar_q_z, mu_p_z, logvar_p_z,
              mu_q_z_next, logvar_q_z_next, z_next, mu_p_z_next, logvar_p_z_next):
    with torch.no_grad():
        pred_loss = - bernoulli(x_next, x_next_recon) \
                    + KL(mu_q_z, logvar_q_z, mu_p_z, logvar_p_z) \
                    - entropy(mu_q_z_next, logvar_q_z_next) \
                    - gaussian(z_next, mu_p_z_next, logvar_p_z_next)

        consis_loss = - entropy(mu_q_z_next, logvar_q_z_next) \
                      - gaussian(z_next, mu_p_z_next, logvar_p_z_next) \
                      + KL(mu_q_z, logvar_q_z, mu_p_z, logvar_p_z)
    return pred_loss, consis_loss