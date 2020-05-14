import torch
import torch.nn.functional as F
import numpy as np

class SimpleOneStepModel:
    """
    Simple one-step model that deterministically predicts s_{t+1}-s_t given s_t, a_t.
    """
    def __init__(self, networks, optimisers, l2_reg=0.0, device="cpu"):
        self.networks = networks
        self.optimisers = optimisers
        self.state_dim = networks[0].state_dim
        self.ac_dim = networks[0].ac_dim
        self.l2_reg = l2_reg
        self.device = device

    def predict(self, state, action, osm_ind=0, normed_input=False):
        """Make a prediction given the current state of the model"""
        self.networks[osm_ind].eval()
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        if len(action.shape) == 1:
            action = action.reshape(1, -1)
        if normed_input:
            state_s = torch.tensor(state, dtype=torch.float32, device=self.device)
            state = self.networks[0].state_scaler.inverse_transform(state)
        else:
            state_s = torch.tensor(self.networks[0].state_scaler.transform(state), dtype=torch.float32, device=self.device)
        action = torch.tensor(action, dtype=torch.float32, device=self.device)
        delta_s = self.networks[osm_ind](state_s, action).detach()
        delta_s_unnorm = self.networks[0].diff_scaler.inverse_transform(delta_s.cpu().data.numpy())
        delta_s_unnorm = np.clip(delta_s_unnorm, -10.0, 10.0)
        if normed_input:
            try:
                return self.networks[0].state_scaler.transform(state + delta_s_unnorm)
            except:
                import joblib
                joblib.dump((self.networks[0].state_scaler, state, delta_s_unnorm), "state_scaler_error.pkl")
                return self.networks[0].state_scaler.transform(state + delta_s_unnorm + 0.01) #sometimes get weird error here...
        else:
            return state + delta_s_unnorm

    def get_errors(self, states, actions, next_states, osm_ind=0):
        self.networks[osm_ind].eval()
        diff = next_states - states
        states_norm = torch.tensor(self.networks[0].state_scaler.transform(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        diff_norm = torch.tensor(self.networks[0].diff_scaler.transform(diff), dtype=torch.float32, device=self.device)
        pred_diff = self.networks[osm_ind](states_norm, actions).detach()
        return F.mse_loss(pred_diff, diff_norm)

    def train_on_batch(self, states, actions, next_states, osm_ind):
        """Train our model on a batch of s_t, a_t, s_{t+1}"""
        self.networks[osm_ind].train()
        diff = next_states - states
        states = torch.tensor(self.networks[0].state_scaler.transform(states), dtype=torch.float32, device=self.device)
        diff = torch.tensor(self.networks[0].diff_scaler.transform(diff), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        self.optimisers[osm_ind].zero_grad()
        pred_diffs = self.networks[osm_ind](states, actions)
        loss = F.mse_loss(pred_diffs, diff)
        loss_reg = 0
        for p in self.networks[osm_ind].parameters():
            loss_reg += torch.dot(p.view(-1), p.view(-1))
        loss_tot = loss + self.l2_reg * loss_reg
        loss_tot.backward()
        self.optimisers[osm_ind].step()
        return loss.item(), self.l2_reg * loss_reg.item()

    def fit_scalers(self, states, state_diffs, noise=[0.05, 0.01], env=None):
        if states is not None:
            self.networks[0].state_scaler.fit(states+noise[0]*np.random.randn(*states.shape))
        if state_diffs is not None:
            self.networks[0].diff_scaler.fit(state_diffs + noise[1]*np.random.randn(*state_diffs.shape))

    def load(self, state_dicts, state_scaler, diff_scaler):
        if state_dicts is not None:
            for i, dict in enumerate(state_dicts):
                self.networks[i].load_state_dict(dict)
        if state_scaler is not None:
            self.networks[0].state_scaler = state_scaler
        if diff_scaler is not None:
            self.networks[0].diff_scaler = diff_scaler

    def save(self):
        return [net.state_dict() for net in self.networks], self.networks[0].state_scaler, self.networks[0].diff_scaler