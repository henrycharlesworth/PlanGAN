import torch
import torch.nn as nn
import torch.nn.functional as F
from full_algorithm.spectral_norm import SpectralNorm

class DynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=[512, 512]):
        super(DynamicsModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        layers = [nn.Linear(state_dim+action_dim, hidden_sizes[0]), nn.ReLU()]
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_sizes[-1], state_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, states, actions):
        input = torch.cat((states, actions), dim=-1)
        return self.model(input)


class Generator_FC(nn.Module):
    def __init__(self, state_dim, goal_dim, latent_dim, out_dim, hidden_sizes=[512, 512],
                 batch_norm=True, skip_conns=True, device="cpu"):
        super(Generator_FC, self).__init__()
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        self.batch_norm = batch_norm
        self.skip_conns = skip_conns
        self.device = device

        def block(in_feat, out_feat):
            layers = [nn.Linear(in_feat, out_feat)]
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_feat))
            layers.append(nn.ReLU())
            return layers

        input_dim = state_dim + goal_dim + latent_dim
        if skip_conns:
            cond_dim = state_dim + goal_dim
        else:
            cond_dim = 0

        blocks = nn.ModuleList()
        blocks.append(nn.Sequential(*block(input_dim, hidden_sizes[0])))
        for i in range(len(hidden_sizes)-1):
            blocks.append(nn.Sequential(*block(hidden_sizes[i]+cond_dim, hidden_sizes[i+1])))
        self.blocks = blocks

        self.out_layer = nn.Linear(hidden_sizes[-1]+cond_dim, out_dim)

    def forward(self, z, x, g):
        input = torch.cat((z, x, g), dim=-1)
        if self.skip_conns:
            cond = torch.cat((x, g), dim=-1)
        else:
            cond = torch.empty(z.shape[0], 0).to(self.device)
        out = self.blocks[0](input)
        for i in range(1, len(self.blocks)):
            out = self.blocks[i](torch.cat((out, cond), dim=-1))
        out = self.out_layer(torch.cat((out, cond), dim=-1))
        return out


class Discriminator_FC(nn.Module):
    def __init__(self, state_dim, goal_dim, hidden_sizes=[512, 512], spectral_norm=True,
                 skip_conns=True, device="cpu"):
        super(Discriminator_FC, self).__init__()
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.spectral_norm = spectral_norm
        self.skip_conns = skip_conns
        self.device = device

        def block(in_feat, out_feat):
            if spectral_norm:
                layers = [SpectralNorm(nn.Linear(in_feat, out_feat))]
            else:
                layers = [nn.Linear(in_feat, out_feat)]
            layers.append(nn.LeakyReLU())
            return layers

        input_dim = 2*state_dim + goal_dim
        if skip_conns:
            cond_dim = state_dim + goal_dim
        else:
            cond_dim = 0
        blocks = nn.ModuleList()
        blocks.append(nn.Sequential(*block(input_dim, hidden_sizes[0])))
        for i in range(len(hidden_sizes)-1):
            blocks.append(nn.Sequential(*block(hidden_sizes[i]+cond_dim, hidden_sizes[i+1])))
        self.blocks = blocks

        """
        if spectral_norm:
            self.out_layer = SpectralNorm(nn.Linear(hidden_sizes[-1]+cond_dim, 1))
        else:
            self.out_layer = nn.Linear(hidden_sizes[-1]+cond_dim, 1)
        """
        self.out_layer = nn.Linear(hidden_sizes[-1] + cond_dim, 1)

    def forward(self, x, xf, g):
        input = torch.cat((x, xf, g), dim=-1)
        if self.skip_conns:
            cond = torch.cat((x, g), dim=-1)
        else:
            cond = torch.empty(x.shape[0], 0).to(self.device)
        out = self.blocks[0](input)
        for i in range(1, len(self.blocks)):
            out = self.blocks[i](torch.cat((out, cond), dim=-1))
        out = self.out_layer(torch.cat((out, cond), dim=-1))
        return out