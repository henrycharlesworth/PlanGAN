"""
Prioritized Experience Replay part adapted from:
https://github.com/rlcode/per/blob/master/SumTree.py
Hindsight Experience Replay style buffer inspired by:
https://github.com/vitchyr/rlkit/blob/master/rlkit/data_management/obs_dict_replay_buffer.py
"""

import numpy as np

class ReplayBuffer(object):
    """
    Replay buffer that allows us to sample
    """
    def __init__(self, capacity, obs_dim, action_dim, goal_dim, tau, image_obs=False, image_goal=False,
                 per=True, e=0.01, a=0.6):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.tau = tau
        self.curr_size = 0
        self.top = 0
        self.per = per

        if per:
            self.e = e
            self.a = a
            self.tree = np.zeros((2*capacity-1),)
        if image_obs:
            obs_dtype = np.uint8
        else:
            obs_dtype = np.float32
        if image_goal:
            goal_dtype = np.uint8
        else:
            goal_dtype = np.float32

        self._observations = np.zeros((capacity, obs_dim), dtype=obs_dtype)
        self._next_observations = np.zeros((capacity, obs_dim), dtype=obs_dtype) #easier to store repeats, despite inefficiency.
        self._achieved_goals = np.zeros((capacity, goal_dim), dtype=goal_dtype)
        self._actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self._idx_to_valid_future_goal = [None] * capacity

    def _propagate(self, idx, change):
        """
        If we are using prioritized experience replay, propagate a change in priority
        """
        parent = (idx-1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def _tree_total(self):
        return self.tree[0]

    def _retrieve(self, idx, s):
        """
        Find sample on leaf node
        """
        left = 2*idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def _update(self, idx, p):
        """
        Update priority of a sample
        """
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def _get_indices(self, s):
        """
        Take value s between 0 and total and retrieve the tree and data indices.
        """
        tree_idx = self._retrieve(0, s)
        data_idx = tree_idx - self.capacity + 1
        return int(tree_idx), int(data_idx)

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
        if self.per:
            errors = path["errors"]
        path_len = observations.shape[0]

        if self.top + path_len > self.capacity:
            """following is for wrapping around when the buffer is full"""
            num_pre_wrap_steps = self.capacity - self.top
            num_post_wrap_steps = path_len - num_pre_wrap_steps
            pre_wrap_buffer_slice = np.s_[self.top:self.top+num_pre_wrap_steps, :]
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
                if i + self.tau >= self.capacity:
                    self._idx_to_valid_future_goal[i] = np.arange((i+self.tau)%self.capacity, num_post_wrap_steps)
                else:
                    self._idx_to_valid_future_goal[i] = np.hstack((
                        np.arange(i+self.tau, self.capacity),
                        np.arange(0, num_post_wrap_steps)
                    ))
            for i in range(0, num_post_wrap_steps):
                self._idx_to_valid_future_goal[i] = np.arange((i+self.tau), num_post_wrap_steps)
        else:
            slc = np.s_[self.top:self.top+path_len, :]
            self._observations[slc] = observations
            self._next_observations[slc] = next_observations
            self._actions[slc] = actions
            self._achieved_goals[slc] = goals
            for i in range(self.top, self.top+path_len):
                self._idx_to_valid_future_goal[i] = np.arange((i+self.tau), self.top+path_len)

        #insert priorities
        if self.per:
            for idx in range(self.top, self.top + path_len):
                data_idx = idx % self.capacity
                tree_idx = data_idx + self.capacity - 1
                self._update(tree_idx, self._get_priority(errors[idx-self.top]))

        self.top = (self.top + path_len) % self.capacity
        self.curr_size = min(self.curr_size+path_len, self.capacity)

    def sample_for_model_training(self, batch_size, noise=[0.0, 0.0, 0.0]):
        if self.per:
            tree_indices = np.zeros((batch_size,)).astype(int)
            data_indices = np.zeros((batch_size,)).astype(int)
            total = self._tree_total()
            for i in range(batch_size):
                s = np.random.uniform(0, total)
                tree_idx, data_idx = self._get_indices(s)
                tree_indices[i] = tree_idx
                data_indices[i] = data_idx
        else:
            data_indices = np.random.randint(0, self.curr_size, batch_size).astype(int)
            tree_indices = None
        obs = self._observations[data_indices, :] + noise[0]*np.random.randn(len(data_indices), self.obs_dim)
        action = np.clip(self._actions[data_indices, :] + noise[1]*np.random.randn(len(data_indices), self.action_dim), -1.0, 1.0)
        next_obs = self._next_observations[data_indices, :] + noise[2]*np.random.randn(len(data_indices), self.obs_dim)

        return obs, action, next_obs, tree_indices

    def update_priorities(self, errors, tree_indices):
        for i in range(len(errors)):
            p = self._get_priority(errors[i])
            self._update(tree_indices[i], p)

    def sample_for_gan(self, batch_size, noise=[0.0, 0.0, 0.0]):
        state_inds = []
        goal_inds = []
        target_inds = []
        if self.per:
            total = self._tree_total()
        while len(state_inds) < batch_size:
            if self.per:
                s = np.random.uniform(0, total)
                tree_idx, data_idx = self._get_indices(s)
            else:
                data_idx = int(np.random.randint(0, self.curr_size))
            if len(self._idx_to_valid_future_goal[data_idx]) == 0:
                continue
            else:
                state_inds.append(data_idx)
                target_inds.append(self._idx_to_valid_future_goal[data_idx][0])
                num_options = len(self._idx_to_valid_future_goal[data_idx])
                goal_inds.append(self._idx_to_valid_future_goal[data_idx][int(np.random.randint(0, num_options))])
        obs = self._observations[state_inds, :] + noise[0]*np.random.randn(len(state_inds), self.obs_dim)
        goals = self._achieved_goals[goal_inds, :] + noise[1]*np.random.randn(len(goal_inds), self.goal_dim)
        targets = self._observations[target_inds, :] + noise[2]*np.random.randn(len(target_inds), self.obs_dim)
        return obs, goals, targets