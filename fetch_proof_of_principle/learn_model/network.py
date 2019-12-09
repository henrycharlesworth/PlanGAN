import torch
import torch.nn as nn
import torch.nn.functional as F

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