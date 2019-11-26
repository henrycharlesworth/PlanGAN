import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from fetch_proof_of_principle.networks.spectral_norm import SpectralNorm
except:
    from networks.spectral_norm import SpectralNorm

class Generator(nn.Module):
    def __init__(self, state_dim, latent_dim, out_dim, hidden_sizes=[256, 128], batch_norm=True, goal_dim=None,
                 cond_on_goal=False, tanh=False, tanh_value=1.0, clamp=False, clamp_value=1.0):
        super(Generator, self).__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.goal_dim = goal_dim
        self.out_dim = out_dim
        self.cond_on_goal = cond_on_goal
        self.tanh = tanh
        self.tanh_value = tanh_value
        self.clamp = False
        self.clamp_value = False

        def block(in_feat, out_feat):
            layers = [nn.Linear(in_feat, out_feat)]
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_feat))
            layers.append(nn.ReLU())
            return layers

        if self.cond_on_goal:
            input_dim = state_dim + goal_dim + latent_dim
        else:
            input_dim = state_dim + latent_dim

        blocks = []
        blocks += block(input_dim, hidden_sizes[0])
        for i in range(len(hidden_sizes)-1):
            blocks += block(hidden_sizes[i], hidden_sizes[i+1])

        self.model = nn.Sequential(
            *blocks,
            nn.Linear(hidden_sizes[-1], out_dim)
        )

    def forward(self, z, x, g=None):
        if self.cond_on_goal:
            input = torch.cat((z, x, g), dim=-1)
        else:
            input = torch.cat((z, x), dim=-1)
        output = self.model(input)
        if self.tanh:
            output = self.tanh_value*torch.tanh(output)
        if self.clamp:
            output = torch.clamp(output, min=-self.clamp_value, max=self.clamp_value)
        return output


class Discriminator(nn.Module):
    def __init__(self, state_dim, goal_dim=None, cond_on_goal=False, hidden_sizes=[256, 128], spectral_norm=False,
                 clamp=False, clamp_value=1.0):
        super(Discriminator, self).__init__()
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.cond_on_goal = cond_on_goal
        self.spectral_norm = spectral_norm
        self.clamp = clamp
        self.clamp_value = clamp_value

        def block(in_feat, out_feat):
            if spectral_norm:
                layers = [SpectralNorm(nn.Linear(in_feat, out_feat))]
            else:
                layers = [nn.Linear(in_feat, out_feat)]
            layers.append(nn.LeakyReLU())
            return layers

        if self.cond_on_goal:
            input_dim = state_dim*2 + goal_dim
        else:
            input_dim = state_dim*2

        blocks = []
        blocks += block(input_dim, hidden_sizes[0])
        for i in range(len(hidden_sizes)-1):
            blocks += block(hidden_sizes[i], hidden_sizes[i+1])

        self.model = nn.Sequential(
            *blocks,
            nn.Linear(hidden_sizes[-1], 1)
        )

    def forward(self, x, xf, g=None):
        if self.cond_on_goal:
            input = torch.cat((x, xf, g), dim=-1)
        else:
            input = torch.cat((x, xf), dim=-1)
        if self.clamp:
            return torch.clamp(self.model(input), min=-self.clamp_value, max=self.clamp_value)
        else:
            return self.model(input)




"""
ALTERNATIVE ARCHITECTURES

try feeding in "label" (state or state+goal) into each layer, including the final.
"""
class Generator_alt(nn.Module):
    def __init__(self, state_dim, latent_dim, out_dim, hidden_sizes=[256, 128], batch_norm=True, goal_dim=None,
                 cond_on_goal=False, tanh=False, tanh_value=1.0, clamp=False, clamp_value=1.0):
        super(Generator_alt, self).__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.goal_dim = goal_dim
        self.out_dim = out_dim
        self.cond_on_goal = cond_on_goal
        self.tanh = tanh
        self.tanh_value = tanh_value
        self.clamp = False
        self.clamp_value=False

        def block(in_feat, out_feat):
            layers = [nn.Linear(in_feat, out_feat)]
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_feat))
            layers.append(nn.ReLU())
            return layers

        if self.cond_on_goal:
            input_dim = state_dim + goal_dim + latent_dim
            cond_dim = state_dim + goal_dim
        else:
            input_dim = state_dim + latent_dim
            cond_dim = state_dim

        blocks = nn.ModuleList()
        blocks.append(nn.Sequential(*block(input_dim, hidden_sizes[0])))
        for i in range(len(hidden_sizes)-1):
            blocks.append(nn.Sequential(*block(hidden_sizes[i] + cond_dim, hidden_sizes[i+1])))

        self.blocks = blocks
        self.out_layer = nn.Linear(hidden_sizes[-1] + cond_dim, out_dim)

    def forward(self, z, x, g=None):
        if self.cond_on_goal:
            input = torch.cat((z, x, g), dim=-1)
            cond = torch.cat((x, g), dim=-1)
        else:
            input = torch.cat((z, x), dim=-1)
            cond = x

        out = self.blocks[0](input)
        for i in range(1, len(self.blocks)):
            out = self.blocks[i](torch.cat((out, cond), dim=-1))
        out = self.out_layer(torch.cat((out, cond), dim=-1))
        if self.tanh:
            out = self.tanh_value * torch.tanh(out)
        if self.clamp:
            output = torch.clamp(out, min=-self.clamp_value, max=self.clamp_value)

        return out

class Discriminator_alt(nn.Module):
    def __init__(self, state_dim, goal_dim=None, cond_on_goal=False, hidden_sizes=[256, 128], spectral_norm=False,
                 clamp=False, clamp_value=1.0):
        super(Discriminator_alt, self).__init__()
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.cond_on_goal = cond_on_goal
        self.spectral_norm = spectral_norm
        self.clamp = clamp
        self.clamp_value = clamp_value

        def block(in_feat, out_feat):
            if spectral_norm:
                layers = [SpectralNorm(nn.Linear(in_feat, out_feat))]
            else:
                layers = [nn.Linear(in_feat, out_feat)]
            layers.append(nn.LeakyReLU())
            return layers

        if self.cond_on_goal:
            input_dim = state_dim*2 + goal_dim
            cond_dim = state_dim + goal_dim
        else:
            input_dim = state_dim*2
            cond_dim = state_dim

        blocks = nn.ModuleList()
        blocks.append(nn.Sequential(*block(input_dim, hidden_sizes[0])))
        for i in range(len(hidden_sizes)-1):
            blocks.append(nn.Sequential(*block(hidden_sizes[i]+cond_dim, hidden_sizes[i+1])))

        self.blocks = blocks
        self.out_layer = nn.Linear(hidden_sizes[-1] + cond_dim, 1)

    def forward(self, x, xf, g=None):
        if self.cond_on_goal:
            input = torch.cat((x, xf, g), dim=-1)
            cond = torch.cat((x, g), dim=-1)
        else:
            input = torch.cat((x, xf), dim=-1)
            cond = x

        out = self.blocks[0](input)
        for i in range(1, len(self.blocks)):
            out = self.blocks[i](torch.cat((out, cond), dim=-1))
        if self.clamp:
            return torch.clamp(self.out_layer(torch.cat((out, cond), dim=-1)), min=-self.clamp_value, max=self.clamp_value)
        else:
            return self.out_layer(torch.cat((out, cond), dim=-1))