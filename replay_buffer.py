"""
Hindsight Experience Replay style buffer inspired by:
https://github.com/vitchyr/rlkit/blob/master/rlkit/data_management/obs_dict_replay_buffer.py
"""
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity, obs_dim, ac_dim, goal_dim, tau=5, filter_train_batch=False,
                 random_future_goals=False, env_name="None"):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.ac_dim = ac_dim
        self.goal_dim = goal_dim
        self.tau = tau
        self.top = 0
        self.curr_size = 0
        self.env_name = env_name
        self.filter_train_batch = filter_train_batch
        self.random_future_goals = random_future_goals
        obs_dtype = np.float32
        self._observations = np.zeros((capacity, obs_dim), dtype=obs_dtype)
        self._next_observations = np.zeros((capacity, obs_dim), dtype=obs_dtype)  # easier to store repeats, despite inefficiency.
        self._achieved_goals = np.zeros((capacity, goal_dim), dtype=np.float32)
        self._actions = np.zeros((capacity, ac_dim), dtype=np.float32)
        self._idx_to_valid_future_goal = [None] * capacity

    def sample_for_model_training(self, batch_size, noise=[0.0, 0.0, 0.0]):
        data_indices = np.random.randint(0, self.curr_size, batch_size).astype(int)
        tree_indices = None
        obs = self._observations[data_indices, :] + noise[0] * np.random.randn(len(data_indices), self.obs_dim)
        action = np.clip(
            self._actions[data_indices, :] + noise[1] * np.random.randn(len(data_indices), self.ac_dim), -1.0, 1.0)
        next_obs = self._next_observations[data_indices, :] + noise[2] * np.random.randn(len(data_indices),
                                                                                         self.obs_dim)

        return obs, action, next_obs, tree_indices

    def sample_for_gan_training(self, batch_size, noise=[0.0, 0.0, 0.0]):
        traj_inds = np.zeros((batch_size, self.tau+1)).astype(int)
        goal_inds = np.zeros((batch_size, self.tau)).astype(int)
        c = 0
        while c < batch_size:
            start_idx = int(np.random.randint(0, self.curr_size))
            if len(self._idx_to_valid_future_goal[start_idx]) < self.tau:
                continue
            else:
                possible_goals = self._idx_to_valid_future_goal[start_idx]
                possible_idxs = [np.random.randint(0, len(possible_goals))]
                if self.filter_train_batch:
                    start_goal = self._achieved_goals[start_idx, :]
                    end_goal = self._achieved_goals[possible_goals[possible_idxs[0]], :]
                    if self.env_name.startswith("four_rooms"):
                        if np.linalg.norm(start_goal-end_goal) < 0.3:
                            continue
                    else:
                        if np.linalg.norm(start_goal-end_goal) < 0.01:
                            continue #filter out useless examples where we would be "learning" to achieve an end goal we've already achieved.
                if self.random_future_goals:
                    for i in range(self.tau-1):
                        possible_idxs.append(np.random.randint(possible_idxs[-1], len(possible_goals)))
                else:
                    if possible_idxs[0] < self.tau-1:
                        for i in range(1, self.tau):
                            possible_idxs.append(min(possible_idxs[-1]+1, len(possible_goals)-1))
                    else:
                        idx = possible_idxs[0]
                        for i in range(self.tau-1):
                            possible_idxs.append(idx)
                goal_inds[c, :] = possible_goals[possible_idxs]

                if start_idx + self.tau < self.capacity:
                    traj_inds[c, :] = np.arange(start_idx, start_idx+self.tau+1)
                else:
                    n_pre = self.capacity - start_idx
                    n_post = self.tau + 1 - n_pre
                    traj_inds[c, :n_pre] = np.arange(start_idx, self.capacity)
                    traj_inds[c, n_pre:] = np.arange(0, n_post)
                c += 1
        obs = self._observations[np.newaxis, ...][:, traj_inds, :][0]
        actions = self._actions[np.newaxis, ...][:, traj_inds[:, :-1], :][0]
        goals = self._achieved_goals[np.newaxis, ...][:, goal_inds, :][0]
        if noise[0] > 0:
            obs += np.random.randn(*obs.shape)*noise[0]
        if noise[1] > 0:
            actions = np.clip(actions + np.random.randn(*actions.shape)*noise[1], -1.0, 1.0)
        if noise[2] > 0:
            goals += np.random.randn(*goals.shape)*noise[2]
        return obs, actions, goals

    def add_path(self, path):
        """
                Path should be a dictionary with:
                "observations" - np array, path_len * obs_dim
                "next_observations" - no array, path_len * obs_dim
                "actions" - np array, path_len * act_dim
                "achieved_goals" - np array, path_len * goal_dim
                "errors" - np array, path_len - current model errors for predicting next state (if using PER)
                """
        observations = path["observations"]
        actions = path["actions"]
        next_observations = path["next_observations"]
        goals = path["achieved_goals"]
        path_len = observations.shape[0]

        if self.top + path_len > self.capacity:
            """following is for wrapping around when the buffer is full"""
            num_pre_wrap_steps = self.capacity - self.top
            num_post_wrap_steps = path_len - num_pre_wrap_steps
            pre_wrap_buffer_slice = np.s_[self.top:self.top + num_pre_wrap_steps, :]
            pre_wrap_path_slice = np.s_[0:num_pre_wrap_steps, :]
            post_wrap_buffer_slice = np.s_[0:num_post_wrap_steps, :]
            post_wrap_path_slice = np.s_[num_pre_wrap_steps:path_len, :]
            for buffer_slice, path_slice in [(pre_wrap_buffer_slice, pre_wrap_path_slice),
                                             (post_wrap_buffer_slice, post_wrap_path_slice)]:
                self._observations[buffer_slice] = observations[path_slice]
                self._next_observations[buffer_slice] = next_observations[path_slice]
                self._actions[buffer_slice] = actions[path_slice]
                self._achieved_goals[buffer_slice] = goals[path_slice]
            for i in range(self.top, self.capacity):
                if i + 1 >= self.capacity:
                    self._idx_to_valid_future_goal[i] = np.arange((i + 1) % self.capacity, num_post_wrap_steps)
                else:
                    self._idx_to_valid_future_goal[i] = np.hstack((
                        np.arange(i + 1, self.capacity),
                        np.arange(0, num_post_wrap_steps)
                    ))
            for i in range(0, num_post_wrap_steps):
                self._idx_to_valid_future_goal[i] = np.arange((i + 1), num_post_wrap_steps)
        else:
            slc = np.s_[self.top:self.top + path_len, :]
            self._observations[slc] = observations
            self._next_observations[slc] = next_observations
            self._actions[slc] = actions
            self._achieved_goals[slc] = goals
            for i in range(self.top, self.top + path_len):
                self._idx_to_valid_future_goal[i] = np.arange((i + 1), self.top + path_len)

        self.top = (self.top + path_len) % self.capacity
        self.curr_size = min(self.curr_size + path_len, self.capacity)