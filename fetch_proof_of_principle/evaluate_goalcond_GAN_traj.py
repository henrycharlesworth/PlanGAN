import numpy as np
import joblib
import torch

from fetch_proof_of_principle.networks.networks import Generator_alt

state_dim = 3
latent_dim = 64
tau = 5
device = "cuda"
netG = Generator_alt(state_dim, latent_dim, state_dim, batch_norm=True, hidden_sizes=[128, 256, 512], goal_dim=state_dim, cond_on_goal=True).to(device)
dict_G, _ = joblib.load("results/goal_cond/from_start_standard/parameters.pkl")
netG.load_state_dict(dict_G)
netG.eval()

num_trajs = 1000
max_steps = 10
dist_threshold = 0.1
min_successes_to_terminate = 5

start_state = np.array([-0.00934646,  0.63374764, -1.1993216 ])
end_goal = np.array([-0.8914087,  1.8443041,  1.1627929])
goals = torch.tensor(end_goal, dtype=torch.float32, device="cuda").view(1,-1).repeat(num_trajs, 1)

