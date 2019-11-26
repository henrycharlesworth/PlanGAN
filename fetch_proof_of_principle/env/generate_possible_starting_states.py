import numpy as np
from fetch_proof_of_principle.env.fetch_reach_mod import FetchReachMod
from sklearn.cluster import KMeans
from pyflann import FLANN
import joblib

env = FetchReachMod()

lower_limits = env.initial_gripper_xpos - np.array([0.25, 0.25, 0.2])
higher_limits = env.initial_gripper_xpos + np.array([0.25, 0.25, 0.2])

num_steps = 100000

num_start_states = 100

saved_states = []
achieved_goal = np.zeros((num_steps, 3))
c = 0
while c < num_steps:
    #env.render()
    obs, _, _, _ = env.step(env.action_space.sample())
    goal = obs["achieved_goal"]
    if goal[0] < lower_limits[0] or goal[0] > higher_limits[0] or goal[1] < lower_limits[1] or goal[1] > higher_limits[1] or goal[2] < lower_limits[2] or goal[2] > higher_limits[2]:
        env.reset()
        continue
    saved_states.append(env.save_state())
    achieved_goal[c,:] = obs["achieved_goal"]
    c += 1
print("DATA GENERATED")
k_means = KMeans(n_clusters=num_start_states).fit(achieved_goal)
centres = k_means.cluster_centers_
print("K MEANS DONE")

point_lookup = FLANN()
point_lookup.build_index(achieved_goal.astype(float), **{"algorithm": "kmeans", "branching": 32, "iterations": 7})

nearest_point, _ = point_lookup.nn_index(centres, num_neighbors=1)

saved_states = [saved_states[i] for i in nearest_point]
achieved_goal = achieved_goal[nearest_point, :]

joblib.dump(saved_states, "starting_states.pkl")

print("DONE")
