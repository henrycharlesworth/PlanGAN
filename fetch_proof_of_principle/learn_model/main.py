import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler

from fetch_proof_of_principle.env.fetch_reach_mod import FetchReachMod
from fetch_proof_of_principle.learn_model.network import DynamicsModel

num_trajs = 100
num_trajs_val = 100
traj_length = 100
state_dim = 10
ac_dim = 4
device = "cuda" if torch.cuda.is_available() else "cpu"
#device="cpu"

batch_size = 512
train_steps = 100000
l2reg = 0.0001

states = np.zeros((num_trajs, traj_length+1, state_dim))
actions = np.zeros((num_trajs, traj_length, ac_dim))

states_val = np.zeros((num_trajs_val, traj_length+1, state_dim))
actions_val = np.zeros((num_trajs_val, traj_length, ac_dim))

experiment_name = "batch_512_rand_start"+str(num_trajs) + "_" + str(traj_length)

data_filename = "data"+experiment_name+".pkl"
if os.path.exists(data_filename):
    states, actions, states_val, actions_val = joblib.load(data_filename)
else:
    possible_start_states = joblib.load("../starting_states.pkl")
    env = FetchReachMod(random_start=True, possible_start_states=possible_start_states)
    for t in range(num_trajs):
        obs = env.reset()
        states[t, 0, :] = obs["observation"]
        for s in range(traj_length):
            a = env.action_space.sample()
            actions[t, s, :] = a
            obs, _, _, _ = env.step(a)
            states[t, s+1, :] = obs["observation"]
        print("Traj %d generated" % t)

    for t in range(num_trajs_val):
        obs = env.reset()
        states_val[t, 0, :] = obs["observation"]
        for s in range(traj_length):
            a = env.action_space.sample()
            actions_val[t, s, :] = a
            obs, _, _, _ = env.step(a)
            states_val[t, s+1, :] = obs["observation"]
        print("Val traj %d generated" % t)

    joblib.dump((states, actions, states_val, actions_val), data_filename)

state_diff = states[:, 1:, :] - states[:, :-1, :]
state_diff_val = states_val[:, 1:, :] - states_val[:, :-1, :]
state_scaler = StandardScaler()
state_diff_scaler = StandardScaler()
states_norm = state_scaler.fit_transform(states[:,:-1,:].reshape(-1, state_dim)).reshape(num_trajs, traj_length, state_dim)
state_diff_norm = state_diff_scaler.fit_transform(state_diff.reshape(-1, state_dim)).reshape(num_trajs, traj_length, state_dim)

states_norm_val = torch.tensor(state_scaler.transform(states_val[:, :-1, :].reshape(-1, state_dim)), dtype=torch.float32).to(device)
state_diff_norm_val = torch.tensor(state_diff_scaler.transform(state_diff_val.reshape(-1, state_dim)), dtype=torch.float32).to(device)
actions_val = torch.tensor(actions_val.reshape(-1, ac_dim), dtype=torch.float32).to(device)

def sample_batch(size):
    t_ind = np.random.randint(0, num_trajs, size)
    s_ind = np.random.randint(0, traj_length, size)
    return torch.tensor(states_norm[t_ind, s_ind, :], dtype=torch.float32).to(device), \
           torch.tensor(actions[t_ind, s_ind, :], dtype=torch.float32).to(device), \
           torch.tensor(state_diff_norm[t_ind, s_ind, :], dtype=torch.float32).to(device)

model = DynamicsModel(state_dim, ac_dim).to(device)
optimiser_m = torch.optim.Adam(model.parameters(), lr=0.001)
loss_record = {"main": [], "val": []}
"""TRAINING"""
for t in range(train_steps):
    s, a, sd = sample_batch(batch_size)
    pred_state_diffs = model(s, a)

    loss = F.mse_loss(sd, pred_state_diffs)
    loss_reg = 0
    for p in model.parameters():
        loss_reg += torch.dot(p.view(-1), p.view(-1))
    loss_tot = loss + l2reg*loss_reg

    pred_state_diff_val = model(states_norm_val, actions_val).detach()
    loss_val = F.mse_loss(pred_state_diff_val, state_diff_norm_val)

    optimiser_m.zero_grad()
    loss_tot.backward()
    optimiser_m.step()
    print("Training step %d, loss: %f. reg loss: %f val loss: %f" % (t, loss.item(), l2reg*loss_reg.item(), loss_val.item()))
    loss_record["main"].append(loss.item())
    loss_record["val"].append(loss_val.item())

joblib.dump((state_scaler, state_diff_scaler, model.state_dict(), loss_record), experiment_name+"_results.pkl")