from fetch_proof_of_principle.env.fetch_reach_mod import FetchReachMod
import numpy as np

env = FetchReachMod()

actions = np.zeros((10000, 4))

obs = env._get_obs()

for i in range(10000):
    a = env.action_space.sample()
    actions[i, :] = a

print("")