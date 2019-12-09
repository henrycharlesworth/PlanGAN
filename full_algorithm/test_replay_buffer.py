from full_algorithm.replay_buffer import ReplayBuffer
import numpy as np

size = 8
obs_dim = 1
ac_dim = 1
goal_dim = 1
tau = 2

obs = np.random.randn(size, obs_dim)
ac = np.random.randn(size, ac_dim)
next_obs = np.random.randn(size, obs_dim)
goals = np.random.randn(size, goal_dim)
errors = np.random.rand(size)
path = {"observations": obs, "actions": ac, "next_observations": next_obs,
        "achieved_goals": goals, "errors": errors}

buffer = ReplayBuffer(size, obs_dim, ac_dim, goal_dim, tau)
buffer.add_path(path)

"""
p = [buffer._get_priority(error) for error in errors]
tot = np.sum(p)

count = {}
for i in range(size):
    count[i] = 0

for i in range(100000):
    tree_idx, data_idx = buffer._get_indices(np.random.uniform(0, tot))
    count[data_idx] += 1

count = np.array([count[i] for i in range(size)])
count = count / np.sum(count)

p = p/np.sum(p)

errors_2 = np.random.rand(size)
p = [buffer._get_priority(error) for error in errors_2]
tot = np.sum(p)

buffer.update_priorities(errors_2, np.arange(7,15))

count = {}
for i in range(size):
    count[i] = 0

for i in range(100000):
    tree_idx, data_idx = buffer._get_indices(np.random.uniform(0, tot))
    count[data_idx] += 1

count = np.array([count[i] for i in range(size)])
count = count / np.sum(count)

p=p/np.sum(p)
"""

"""
obs_2 = np.random.randn(size-2, obs_dim)
ac_2 = np.random.randn(size-2, ac_dim)
next_obs_2 = np.random.randn(size-2, obs_dim)
goals_2 = np.random.randn(size-2, goal_dim)
errors_2 = np.random.rand(size-2)
path_2 = {"observations": obs_2, "actions": ac_2, "next_observations": next_obs_2,
        "achieved_goals": goals_2, "errors": errors_2}

buffer.add_path(path_2)
"""

#obs_batch, ac_batch, next_obs_batch, tree_indices = buffer.sample_for_model_training(16)
curr_goals, end_goals, target_goals = buffer.sample_for_gan(16)

print("OK?")