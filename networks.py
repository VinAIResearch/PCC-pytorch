import torch
from torch import nn

torch.set_default_dtype(torch.float64)

class Encoder(nn.Module):
    # P(z_t | x_t) and Q(z^_t+1 | x_t+1)
    def __init__(self, net, x_dim, z_dim):
        super(Encoder, self).__init__()
        self.net = net
        self.x_dim = x_dim
        self.z_dim = z_dim

    def forward(self, x):
        # mean and variance of p(z|x)
        return self.net(x).chunk(2, dim = 1)

class Decoder(nn.Module):
    # P(x_t+1 | z^_t+1)
    def __init__(self, net, z_dim, x_dim):
        super(Decoder, self).__init__()
        self.net = net
        self.z_dim = z_dim
        self.x_dim = x_dim

    def forward(self, z):
        return self.net(z)

class Dynamics(nn.Module):
    # P(z^_t+1 | z_t, u_t)
    def __init__(self, net, net_z_next, net_A, net_B, z_dim, u_dim, armotized):
        super(Dynamics, self).__init__()
        self.net = net
        self.net_z_next = net_z_next
        self.net_A = net_A
        self.net_B = net_B
        self.z_dim = z_dim
        self.u_dim = u_dim
        self.armotized = armotized

    def forward(self, z_t, u_t):
        z_u_t = torch.cat((z_t, u_t), dim = 1)
        h = self.net(z_u_t)
        mu, logvar = self.net_z_next(h).chunk(2, dim = 1)
        if self.armotized:
            A = self.net_A(h)
            B = self.net_B(h)
        else:
            A, B = None, None
        return mu, logvar, A, B # skip connection

class BackwardDynamics(nn.Module):
    # Q(z_t | z^_t+1, x_t, u_t)
    def __init__(self, net_z, net_u, net_x, net_joint, z_dim, u_dim, x_dim):
        super(BackwardDynamics, self).__init__()
        self.net_z = net_z
        self.net_u = net_u
        self.net_x = net_x
        self.net_joint = net_joint

        self.z_dim = z_dim
        self.u_dim = u_dim
        self.x_dim = x_dim

    def forward(self, z_t, u_t, x_t):
        z_t_out = self.net_z(z_t)
        u_t_out = self.net_u(u_t)
        x_t_out = self.net_x(x_t)

        z_u_x = torch.cat((z_t_out, u_t_out, x_t_out), dim = 1)
        return self.net_joint(z_u_x).chunk(2, dim = 1)

class PlanarEncoder(Encoder):
    def __init__(self, x_dim = 1600, z_dim = 2):
        net = nn.Sequential(
            nn.Linear(x_dim, 300),
            nn.BatchNorm1d(300),
            nn.ReLU(),

            nn.Linear(300, 300),
            nn.BatchNorm1d(300),
            nn.ReLU(),

            nn.Linear(300, z_dim * 2)
        )
        super(PlanarEncoder, self).__init__(net, x_dim, z_dim)

class PlanarDecoder(Decoder):
    def __init__(self, z_dim = 2, x_dim = 1600):
        net = nn.Sequential(
            nn.Linear(z_dim, 300),
            nn.BatchNorm1d(300),
            nn.ReLU(),

            nn.Linear(300, 300),
            nn.BatchNorm1d(300),
            nn.ReLU(),

            nn.Linear(300, x_dim),
            nn.Sigmoid()
        )
        super(PlanarDecoder, self).__init__(net, z_dim, x_dim)

class PlanarDynamics(Dynamics):
    def __init__(self, armotized, z_dim = 2, u_dim = 2):
        net = nn.Sequential(
            nn.Linear(z_dim + u_dim, 20),
            nn.BatchNorm1d(20),
            nn.ReLU(),

            nn.Linear(20, 20),
            nn.BatchNorm1d(20),
            nn.ReLU()
        )
        net_z_next = nn.Linear(20, z_dim * 2)
        if armotized:
            net_A = nn.Linear(20, z_dim**2)
            net_B = nn.Linear(20, u_dim**2)
        else:
            net_A, net_B = None, None
        super(PlanarDynamics, self).__init__(net, net_z_next, net_A, net_B, z_dim, u_dim, armotized)

class PlanarBackwardDynamics(BackwardDynamics):
    def __init__(self, z_dim, u_dim, x_dim):
        net_z = nn.Linear(z_dim, 5)
        net_u = nn.Linear(u_dim, 5)
        net_x = nn.Linear(x_dim, 100)
        net_joint = nn.Sequential(
            nn.Linear(5 + 5 + 100, 100),
            nn.ReLU(),

            nn.Linear(100, z_dim * 2)
        )
        super(PlanarBackwardDynamics, self).__init__(net_z, net_u, net_x, net_joint, z_dim, u_dim, x_dim)

# class PendulumEncoder(Encoder):
#     def __init__(self, obs_dim = 4608, z_dim = 3):
#         net = nn.Sequential(
#             nn.Linear(obs_dim, 800),
#             nn.BatchNorm1d(800),
#             nn.ReLU(),

#             nn.Linear(800, 800),
#             nn.BatchNorm1d(800),
#             nn.ReLU(),

#             nn.Linear(800, z_dim * 2)
#         )
#         super(PendulumEncoder, self).__init__(net, obs_dim, z_dim)

# class PendulumDecoder(Decoder):
#     def __init__(self, z_dim = 3, obs_dim = 4608):
#         net = nn.Sequential(
#             nn.Linear(z_dim, 800),
#             nn.BatchNorm1d(800),
#             nn.ReLU(),

#             nn.Linear(800, 800),
#             nn.BatchNorm1d(800),
#             nn.ReLU(),

#             nn.Linear(800, obs_dim)
#         )
#         super(PendulumDecoder, self).__init__(net, z_dim, obs_dim)

# class PendulumTransition(Transition):
#     def __init__(self, z_dim = 3, u_dim = 1):
#         net = nn.Sequential(
#             nn.Linear(z_dim, 100),
#             nn.BatchNorm1d(100),
#             nn.ReLU(),

#             nn.Linear(100, 100),
#             nn.BatchNorm1d(100),
#             nn.ReLU()
#         )
#         super(PendulumTransition, self).__init__(net, z_dim, u_dim)

CONFIG = {
    'planar': (PlanarEncoder, PlanarDecoder, PlanarDynamics, PlanarBackwardDynamics),
    # 'pendulum': (PendulumEncoder, PendulumDecoder, PendulumTransition)
}

def load_config(name):
    return CONFIG[name]

__all__ = ['load_config']