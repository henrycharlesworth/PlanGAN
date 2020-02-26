import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from spectral_norm import SpectralNorm

class OneStepModelFC(nn.Module):
    def __init__(self, state_dim, ac_dim, hidden_sizes=[512, 512], device="cuda"):
        super(OneStepModelFC, self).__init__()
        self.state_dim = state_dim
        self.ac_dim = ac_dim
        self.device = device

        self.state_scaler = StandardScaler()
        self.diff_scaler = StandardScaler()

        layers = [nn.Linear(state_dim + ac_dim, hidden_sizes[0]), nn.ReLU()]
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_sizes[-1], state_dim))
        self.model = nn.Sequential(*layers)
        self.minmax=False

    """
    def fit_scalers(self, states, state_diffs, noise=[0.05, 0.01]):
        if states is not None:
            self.state_scaler.fit(states+noise[0]*np.random.randn(*states.shape))
        if state_diffs is not None:
            self.diff_scaler.fit(state_diffs + noise[1]*np.random.randn(*state_diffs.shape))
    """

    def scaler_transform(self, tensor, name, inverse=False):
        size = tensor.size()
        if name == "state":
            scale = self.state_scaler.scale_
            if self.minmax:
                mean = self.state_scaler.min_
            else:
                mean = self.state_scaler.mean_
        elif name == "diff":
            scale = self.diff_scaler.scale_
            if self.minmax:
                mean = self.diff_scaler.min_
            else:
                mean = self.diff_scaler.mean_
        else:
            raise RuntimeError()
        if len(size) == 2:
            scale = torch.tensor(scale[np.newaxis, ...], dtype=torch.float32, device=self.device)
            mean = torch.tensor(mean[np.newaxis, ...], dtype=torch.float32, device=self.device)
        elif len(size) == 3:
            scale = torch.tensor(scale[np.newaxis, np.newaxis, ...], dtype=torch.float32, device=self.device)
            mean = torch.tensor(mean[np.newaxis, np.newaxis, ...], dtype=torch.float32, device=self.device)
        else:
            raise RuntimeError()
        if inverse:
            return scale*tensor + mean
        else:
            return (tensor - mean) / (scale + 1e-8)

    def forward(self, states, actions):
        input = torch.cat((states, actions), dim=-1)
        return self.model(input)


class OneStepGeneratorFC(nn.Module):
    def __init__(self, state_dim, ac_dim, goal_dim, latent_dim, hidden_sizes=[512, 512],
                 batch_norm=True, output_state_directly=True, output_new_z=False):
        super(OneStepGeneratorFC, self).__init__()
        self.state_dim = state_dim
        self.ac_dim = ac_dim
        self.goal_dim = goal_dim
        self.latent_dim = latent_dim
        self.output_state_directly = output_state_directly
        self.output_new_z = output_new_z
        self.input_dim = state_dim + goal_dim + latent_dim
        self.out_dim = ac_dim
        if output_state_directly:
            self.out_dim += self.state_dim
        if output_new_z:
            self.out_dim += self.latent_dim

        def block(in_feat, out_feat):
            layers = [nn.Linear(in_feat, out_feat)]
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_feat))
            layers.append(nn.ReLU())
            return layers

        self.model = nn.ModuleList()
        self.model.append(nn.Sequential(*block(self.input_dim, hidden_sizes[0])))
        for i in range(len(hidden_sizes) - 1):
            self.model.append(nn.Sequential(*block(hidden_sizes[i], hidden_sizes[i + 1])))
        self.out_layer = nn.Linear(hidden_sizes[-1], self.out_dim)

    def forward(self, z, x, g):
        input = torch.cat((z, x, g), dim=-1)
        for i in range(len(self.model)):
            input = self.model[i](input)
        if self.output_state_directly and self.output_new_z:
            actions, states, new_z = torch.split_with_sizes(self.out_layer(input), [self.ac_dim, self.state_dim, self.latent_dim], dim=-1)
            return torch.tanh(actions), states, new_z
        elif self.output_state_directly:
            actions, states = torch.split_with_sizes(self.out_layer(input), [self.ac_dim, self.state_dim], dim=-1)
            return torch.tanh(actions), states, None
        elif self.output_new_z:
            actions, new_z = torch.split_with_sizes(self.out_layer(input), [self.ac_dim, self.latent_dim], dim=-1)
            return torch.tanh(actions), None, new_z
        else:
            actions = self.out_layer(input)
            return torch.tanh(actions), None, None


class TrajGeneratorFC(nn.Module):
    def __init__(self, state_dim, ac_dim, goal_dim, latent_dim, tau, hidden_sizes=[512, 512],
                 batch_norm=True, use_osm=False, gen_next_z=True, use_same_z=False, device="cuda"):
        super(TrajGeneratorFC, self).__init__()
        self.state_dim = state_dim
        self.ac_dim = ac_dim
        self.goal_dim = goal_dim
        self.latent_dim = latent_dim
        self.tau = tau
        self.gen_next_z = gen_next_z
        self.use_same_z = use_same_z
        self.device = device
        self.one_step_generator = OneStepGeneratorFC(state_dim, ac_dim, goal_dim, latent_dim, hidden_sizes,
                                                     batch_norm=batch_norm, output_state_directly=(1-use_osm),
                                                     output_new_z=gen_next_z).to(device)

    def forward(self, z, x, g, num_steps=None, ac_noise=0.0, state_noise=0.0):
        if num_steps is None:
            num_steps = self.tau
        gen_states = torch.zeros(x.size()[0], num_steps+1, self.state_dim, dtype=torch.float32, device=self.device)
        gen_actions = torch.zeros(x.size()[0], num_steps, self.ac_dim, dtype=torch.float32, device=self.device)
        z_all = torch.zeros(x.size()[0], num_steps, self.latent_dim, dtype=torch.float32, device=self.device)
        gen_states[:, 0, :] = x + state_noise*torch.randn_like(x)
        z_all[:, 0, :] = z
        if len(g.shape) == 2:
            g = g.unsqueeze(0).repeat(num_steps, 1, 1).transpose(0,1)
        for ts in range(num_steps):
            actions, states, new_z = self.one_step_generator(z, x, g[:, ts, :])
            actions = torch.clamp(actions + ac_noise*torch.randn_like(actions), -1.0, 1.0)
            states[..., 9:] = states[..., 9:] + state_noise*torch.randn_like(states[..., 9:])
            x = states
            if self.gen_next_z:
                z = new_z
            elif self.use_same_z:
                z = z
            else:
                z = torch.randn_like(z, dtype=torch.float32, device=self.device)
            gen_states[:, ts+1, :] = x
            gen_actions[:, ts, :] = actions
            if ts < num_steps-1:
                z_all[:, ts+1, :] = z
        return gen_states, gen_actions, z_all


class DiscriminatorFC(nn.Module):
    def __init__(self, state_dim, goal_dim, ac_dim, hidden_sizes=[512, 512]):
        super(DiscriminatorFC, self).__init__()
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.ac_dim = ac_dim
        self.input_dim = 2*state_dim + goal_dim + ac_dim

        def block(in_feat, out_feat, spectral_norm=True):
            if spectral_norm:
                layers = [SpectralNorm(nn.Linear(in_feat, out_feat))]
            else:
                layers = [nn.Linear(in_feat, out_feat)]
            layers.append(nn.LeakyReLU())
            return layers
        self.model = nn.ModuleList()
        self.model.append(nn.Sequential(*block(self.input_dim, hidden_sizes[0])))
        for i in range(len(hidden_sizes)-1):
            self.model.append(nn.Sequential(*block(hidden_sizes[i], hidden_sizes[i+1])))
        self.out_layer = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, x, g, xtau, a):
        input = torch.cat((x, g, xtau, a), dim=-1)
        for i in range(len(self.model)):
            input = self.model[i](input)
        return self.out_layer(input)