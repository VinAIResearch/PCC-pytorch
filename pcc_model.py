import torch
from torch import nn
from networks import *

torch.set_default_dtype(torch.float64)
# torch.manual_seed(0)

class PCC(nn.Module):
    def __init__(self, armotized, x_dim, z_dim, u_dim, env = 'planar'):
        super(PCC, self).__init__()
        enc, dec, dyn, back_dyn = load_config(env)

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.u_dim = u_dim
        self.armotized = armotized

        self.encoder = enc(x_dim, z_dim)
        self.decoder = dec(z_dim, x_dim)
        self.dynamics = dyn(armotized, z_dim, u_dim)
        self.backward_dynamics = back_dyn(z_dim, u_dim, x_dim)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def transition(self, z, u):
        return self.dynamics(z, u)

    def back_dynamics(self, z, u, x):
        return self.backward_dynamics(z, u, x)

    def reparam(self, mean, logvar):
        sigma = (logvar / 2).exp()
        epsilon = torch.randn_like(sigma)
        return mean + torch.mul(epsilon, sigma)

    def forward(self, x, u, x_next):
        # prediction and consistency loss
        # 1st term and 3rd
        mu_q_z_next, logvar_q_z_next = self.encode(x_next) # Q(z^_t+1 | x_t+1)
        z_next = self.reparam(mu_q_z_next, logvar_q_z_next) # sample z^_t+1
        x_next_recon = self.decode(z_next) # P(x_t+1 | z^t_t+1)
        # 2nd term
        mu_q_z, logvar_q_z = self.back_dynamics(z_next, u, x) # Q(z_t | z^_t+1, u_t, x_t)
        mu_p_z, logvar_p_z = self.encode(x) # P(z_t | x_t)

        # 4th term
        z_q = self.reparam(mu_q_z, logvar_q_z) # samples from Q(z_t | z^_t+1, u_t, x_t)
        mu_p_z_next, logvar_p_z_next, _, _ = self.transition(z_q, u) # P(z^_t+1 | z_t, u _t)

        # additional VAE loss
        z_p = self.reparam(mu_p_z, logvar_p_z) # samples from P(z_t | x_t)
        x_recon = self.decode(z_p) # for additional vae loss

        # additional deterministic loss
        mu_z_next_determ, _, A, B = self.transition(mu_p_z, u)
        x_next_determ = self.decode(mu_z_next_determ)

        return x_next_recon.view(x_next_recon.size(0), -1), \
                mu_q_z, logvar_q_z, mu_p_z, logvar_p_z, \
                mu_q_z_next, logvar_q_z_next, \
                z_next, mu_p_z_next, logvar_p_z_next, \
                z_p, u, x_recon.view(x_recon.size(0), -1), x_next_determ.view(x_next_determ.size(0), -1)

    def predict(self, x, u):
        mu, logvar = self.encoder(x)
        z = self.reparam(mu, logvar)
        x_recon = self.decode(z)

        mu_next, logvar_next, A, B = self.transition(z, u)
        z_next = self.reparam(mu_next, logvar_next)
        x_next_pred = self.decode(z_next)
        return x_recon, x_next_pred

# def reparam(mean, logvar):
#     sigma = (logvar / 2).exp()
#     epsilon = torch.randn_like(sigma)
#     return mean + torch.mul(epsilon, sigma)

# def jacobian_1(dynamics, z, u):
#     """
#     compute the jacobian of F(z,u) w.r.t z, u
#     """
#     z_dim, u_dim = z.size(1), u.size(1)
#     z, u = z.squeeze().repeat(z_dim, 1), u.squeeze().repeat(z_dim, 1)
#     z = z.detach().requires_grad_(True)
#     u = u.detach().requires_grad_(True)
#     z_next, _, _, _ = dynamics(z, u)
#     grad_inp = torch.eye(z_dim)
#     A = torch.autograd.grad(z_next, z, grad_inp, retain_graph=True)[0]
#     B = torch.autograd.grad(z_next, u, grad_inp, retain_graph=True)[0]
#     return A, B

# def jacobian_2(dynamics, z, u):
#     """
#     compute the jacobian of F(z,u) w.r.t z, u
#     """
#     z_dim, u_dim = z.size(1), u.size(1)
#     z = z.detach().requires_grad_(True)
#     u = u.detach().requires_grad_(True)
#     z_next, _, _, _ = dynamics(z, u)
#     A = torch.empty(size=(z_dim, z_dim))
#     B = torch.empty(size=(z_dim, u_dim))
#     for i in range(A.size(0)): # for each row
#         grad_inp = torch.zeros(size=(1, A.size(0)))
#         grad_inp[0][i] = 1
#         A[i] = torch.autograd.grad(z_next, z, grad_inp, retain_graph=True)[0]
#     for i in range(B.size(0)): # for each row
#         grad_inp = torch.zeros(size=(1, B.size(0)))
#         grad_inp[0][i] = 1
#         B[i] = torch.autograd.grad(z_next, u, grad_inp, retain_graph=True)[0]
#     return A, B

# enc, dec, dyn, back_dyn = load_config('planar')
# dynamics = dyn(armotized=False, z_dim=2, u_dim=2)
# dynamics.eval()

# import torch.optim as optim
# optimizer = optim.Adam(dynamics.parameters(), betas=(0.9, 0.999), eps=1e-8, lr=0.001)

# z = torch.randn(size=(1, 2))
# z.requires_grad = True
# u = torch.randn(size=(1, 2))
# u.requires_grad = True

# eps_z = torch.normal(0.0, 0.1, size=z.size())
# eps_u = torch.normal(0.0, 0.1, size=u.size())

# mean, logvar, _, _ = dynamics(z, u)
# grad_z = torch.autograd.grad(mean, z, grad_outputs=eps_z, retain_graph=True, create_graph=True)
# grad_u = torch.autograd.grad(mean, u, grad_outputs=eps_u, retain_graph=True, create_graph=True)
# print ('AAA')
# print (grad_z, grad_u)
# A, B = jacobian_1(dynamics, z, u)
# grad_z, grad_u = eps_z.mm(A), eps_u.mm(B)
# print ('BBBB')
# print (grad_z, grad_u)
# A, B = jacobian_1(dynamics, z, u)
# grad_z, grad_u = eps_z.mm(A), eps_u.mm(B)
# print ('BBBB')
# print (grad_z, grad_u)