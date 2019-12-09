import torch
import numpy as np
import joblib
import os

from fetch_proof_of_principle.learn_model.network import DynamicsModel
from fetch_proof_of_principle.env.fetch_reach_mod import FetchReachMod

state_dim = 10
ac_dim = 4
device = "cuda" if torch.cuda.is_available() else "cpu"
num_trajs = 1000
traj_length = 100

model = DynamicsModel(state_dim, ac_dim).to(device)
state_scaler, state_diff_scaler, model_dict, _ = joblib.load(str(num_trajs)+"_"+str(traj_length)+"_results.pkl")
model.load_state_dict(model_dict)
model.eval()

#gather some test trajectories
data_filename = "data_test"+str(num_trajs)+"_"+str(traj_length)+".pkl"
if os.path.exists(data_filename):
    states, actions = joblib.load(data_filename)
else:
    states = np.zeros((num_trajs, traj_length+1, state_dim))
    actions = np.zeros((num_trajs, traj_length, ac_dim))
    env = FetchReachMod()
    for t in range(num_trajs):
        obs = env.reset()
        states[t, 0, :] = obs["observation"]
        for s in range(traj_length):
            a = env.action_space.sample()
            actions[t, s, :] = a
            obs, _, _, _ = env.step(a)
            states[t, s + 1, :] = obs["observation"]
        print("Traj %d generated" % t)
    joblib.dump((states, actions), data_filename)

states_norm = state_scaler.transform(states.reshape(-1, state_dim)).reshape(num_trajs, traj_length+1, state_dim)
avg_state_diff_t = np.zeros((traj_length, state_dim))

states_pred = np.zeros_like(states)
states_pred[:,0,:] = states[:,0,:]

for t in range(traj_length):

    states_in = state_scaler.transform(states_pred[:,t,:])
    in_s = torch.tensor(states_in, dtype=torch.float32).to(device)
    in_a = torch.tensor(actions[:, t, :].reshape(-1, ac_dim), dtype=torch.float32).to(device)
    pred_ds = model(in_s, in_a)
    pred_ds_unnorm = state_diff_scaler.inverse_transform(pred_ds.cpu().data.numpy())
    pred_ds_unnorm = pred_ds_unnorm.reshape(num_trajs, state_dim)
    states_pred[:,t+1,:] = states_pred[:,t,:] + pred_ds_unnorm
    avg_state_diff_t[t, :] = np.mean(np.abs(states_pred[:,t+1,:].reshape(-1, state_dim) - states[:,t+1,:].reshape(-1, state_dim)), axis=0)

print("OK?")