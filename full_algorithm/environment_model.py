import torch
import torch.nn.functional as F
import abc
from sklearn.preprocessing import StandardScaler

class EnvironmentModel(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def predict(self, state, action):
        pass

    @abc.abstractmethod
    def train_on_batch(self, states, actions, next_states):


class SimpleEnvironmentModel(EnvironmentModel):
    def __init__(self, network, l2_reg=0.0001, lr=0.001, betas=(0.9, 0.999), device="cpu"):
        self.network = network
        self.optimiser = torch.optim.Adam(self.network.parameters(), lr=lr, betas=betas)
        self.state_dim = network.state_dim
        self.action_dim = network.action_dim
        self.device = device
        self.l2_reg = l2_reg
        self.state_scaler = StandardScaler()
        self.diff_scaler = StandardScaler()

    def fit_scalers(self, states, state_diffs):
        if states is not None:
            self.state_scaler.fit(states)
        if state_diffs is not None:
            self.diff_scaler.fit(state_diffs)

    def load_params(self, state_dict):
        self.network.load_state_dict(state_dict)

    def get_params(self):
        return self.network.state_dict()

    def predict(self, state, action):
        state_s = torch.tensor(self.state_scaler.transform(state), dtype=torch.float32, device=self.device)
        action = torch.tensor(action, dtype=torch.float32, device=self.device)
        delta_s = self.network(state_s, action).detach()
        delta_s_unnorm = self.diff_scaler.inverse_transform(delta_s.cpu().data.numpy())
        return state + delta_s_unnorm

    def train_on_batch(self, states, actions, next_states):
        diff = next_states - states
        states = torch.tensor(self.state_scaler.transform(states), dtype=torch.float32, device=self.device)
        diff = torch.tensor(self.diff_scaler.transform(diff), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.float32, device=self.device)

        self.optimiser.zero_grad()
        pred_diffs = self.network(states, actions)
        loss = F.mse_loss(pred_diffs, diff)
        loss_reg = 0
        for p in self.network.parameters():
            loss_reg += torch.dot(p.view(-1), p.view(-1))
        loss_tot = loss_reg + self.l2_reg*loss_reg
        loss_tot.backward()
        self.optimiser.step()
        return loss.item(), loss_reg.item()