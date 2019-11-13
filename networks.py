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
        return self.net(x).chunk(2, dim = -1)

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
        z_u_t = torch.cat((z_t, u_t), dim = -1)
        h = self.net(z_u_t)
        mu, logvar = self.net_z_next(h).chunk(2, dim = -1)
        if self.armotized:
            A = self.net_A(h)
            B = self.net_B(h)
        else:
            A, B = None, None
        return mu + z_t, logvar, A, B # skip connection

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

        z_u_x = torch.cat((z_t_out, u_t_out, x_t_out), dim = -1)
        return self.net_joint(z_u_x).chunk(2, dim = -1)

class PlanarEncoder(Encoder):
    def __init__(self, x_dim = 1600, z_dim = 2):
        net = nn.Sequential(
            nn.Linear(x_dim, 300),
            # nn.BatchNorm1d(300),
            nn.ReLU(),

            nn.Linear(300, 300),
            # nn.BatchNorm1d(300),
            nn.ReLU(),

            nn.Linear(300, z_dim * 2)
        )
        super(PlanarEncoder, self).__init__(net, x_dim, z_dim)

class PlanarDecoder(Decoder):
    def __init__(self, z_dim = 2, x_dim = 1600):
        net = nn.Sequential(
            nn.Linear(z_dim, 300),
            # nn.BatchNorm1d(300),
            nn.ReLU(),

            nn.Linear(300, 300),
            # nn.BatchNorm1d(300),
            nn.ReLU(),

            nn.Linear(300, x_dim),
            nn.Sigmoid()
        )
        super(PlanarDecoder, self).__init__(net, z_dim, x_dim)

class PlanarDynamics(Dynamics):
    def __init__(self, armotized, z_dim = 2, u_dim = 2):
        net = nn.Sequential(
            nn.Linear(z_dim + u_dim, 20),
            # nn.BatchNorm1d(20),
            nn.ReLU(),

            nn.Linear(20, 20),
            # nn.BatchNorm1d(20),
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
    def __init__(self, z_dim=2, u_dim=2, x_dim=1600):
        net_z = nn.Linear(z_dim, 5)
        net_u = nn.Linear(u_dim, 5)
        net_x = nn.Linear(x_dim, 100)
        net_joint = nn.Sequential(
            nn.Linear(5 + 5 + 100, 100),
            nn.ReLU(),

            nn.Linear(100, z_dim * 2)
        )
        super(PlanarBackwardDynamics, self).__init__(net_z, net_u, net_x, net_joint, z_dim, u_dim, x_dim)

class PendulumEncoder(Encoder):
    def __init__(self, x_dim = 4608, z_dim = 3):
        net = nn.Sequential(
            nn.Linear(x_dim, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),

            nn.Linear(500, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),

            nn.Linear(500, z_dim * 2)
        )
        super(PendulumEncoder, self).__init__(net, x_dim, z_dim)

class PendulumDecoder(Decoder):
    def __init__(self, z_dim = 3, x_dim = 4608):
        net = nn.Sequential(
            nn.Linear(z_dim, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),

            nn.Linear(500, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),

            nn.Linear(500, x_dim),
            nn.Sigmoid()
        )
        super(PendulumDecoder, self).__init__(net, z_dim, x_dim)

class PendulumDynamics(Dynamics):
    def __init__(self, armotized, z_dim = 3, u_dim = 1):
        net = nn.Sequential(
            nn.Linear(z_dim + u_dim, 30),
            nn.BatchNorm1d(30),
            nn.ReLU(),

            nn.Linear(30, 30),
            nn.BatchNorm1d(30),
            nn.ReLU()
        )
        net_z_next = nn.Linear(30, z_dim * 2)
        if armotized:
            net_A = nn.Linear(30, z_dim*z_dim)
            net_B = nn.Linear(30, u_dim*z_dim)
        else:
            net_A, net_B = None, None
        super(PendulumDynamics, self).__init__(net, net_z_next, net_A, net_B, z_dim, u_dim, armotized)

class PendulumBackwardDynamics(BackwardDynamics):
    def __init__(self, z_dim=3, u_dim=1, x_dim=4608):
        net_z = nn.Linear(z_dim, 10)
        net_u = nn.Linear(u_dim, 10)
        net_x = nn.Linear(x_dim, 200)
        net_joint = nn.Sequential(
            nn.Linear(10 + 10 + 200, 200),
            nn.ReLU(),

            nn.Linear(200, z_dim * 2)
        )
        super(PendulumBackwardDynamics, self).__init__(net_z, net_u, net_x, net_joint, z_dim, u_dim, x_dim)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size()[0], -1)

class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

class CartPoleEncoder(Encoder):
    def __init__(self, x_dim=(2, 80, 80), z_dim=8):
        x_channels = x_dim[0]
        net = nn.Sequential(
            nn.Conv2d(in_channels=x_channels, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(10),
            nn.ReLU(),

            Flatten(),

            nn.Linear(10*10*10, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),

            nn.Linear(200, z_dim*2)
        )
        super(CartPoleEncoder, self).__init__(net, x_dim, z_dim)

class CartPoleDecoder(Decoder):
    def __init__(self, z_dim=8, x_dim=(2, 80, 80)):
        x_channels = x_dim[0]
        net = nn.Sequential(
            nn.Linear(z_dim, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),

            nn.Linear(200, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),

            View((-1, 10, 10, 10)),

            nn.ConvTranspose2d(in_channels=10, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=32, out_channels=x_channels, kernel_size=5, stride=1, padding=2),
            nn.Sigmoid()
        )
        super(CartPoleDecoder, self).__init__(net, z_dim, x_dim)

class CartPoleDynamics(Dynamics):
    def __init__(self, armotized, z_dim=8, u_dim=1):
        net = nn.Sequential(
            nn.Linear(z_dim + u_dim, 40),
            nn.BatchNorm1d(40),
            nn.ReLU(),

            nn.Linear(40, 40),
            nn.BatchNorm1d(40),
            nn.ReLU()
        )
        net_z_next = nn.Linear(40, z_dim * 2)
        if armotized:
            net_A = nn.Linear(40, z_dim * z_dim)
            net_B = nn.Linear(40, u_dim * z_dim)
        else:
            net_A, net_B = None, None
        super(CartPoleDynamics, self).__init__(net, net_z_next, net_A, net_B, z_dim, u_dim, armotized)

class CartPoleBackwardDynamics(BackwardDynamics):
    def __init__(self, z_dim=8, u_dim=1, x_dim=(2, 80, 80)):
        net_z = nn.Linear(z_dim, 10)
        net_u = nn.Linear(u_dim, 10)

        net_x = nn.Sequential(
            Flatten(),
            nn.Linear(x_dim[0] * x_dim[1] * x_dim[2], 300)
        )
        net_joint = nn.Sequential(
            nn.Linear(10 + 10 + 300, 300),
            nn.ReLU(),

            nn.Linear(300, z_dim * 2)
        )
        super(CartPoleBackwardDynamics, self).__init__(net_z, net_u, net_x, net_joint, z_dim, u_dim, x_dim)

CONFIG = {
    'planar': (PlanarEncoder, PlanarDecoder, PlanarDynamics, PlanarBackwardDynamics),
    'pendulum': (PendulumEncoder, PendulumDecoder, PendulumDynamics, PendulumBackwardDynamics),
    'cartpole': (CartPoleEncoder, CartPoleDecoder, CartPoleDynamics, CartPoleBackwardDynamics)
}

def load_config(name):
    return CONFIG[name]

__all__ = ['load_config']

# device = torch.device("cuda")
# cartpole_encoder = CartPoleEncoder()
# cartpole_encoder.to(device)
# # cartpole_encoder.net[0].to(device)
# print (next(cartpole_encoder.net[0].parameters()).is_cuda)