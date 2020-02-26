"""
Prioritized Experience Replay part adapted from:
https://github.com/rlcode/per/blob/master/SumTree.py
Hindsight Experience Replay style buffer inspired by:
https://github.com/vitchyr/rlkit/blob/master/rlkit/data_management/obs_dict_replay_buffer.py
"""
import numpy as np
from gym.envs.robotics.rotations import quat_mul, quat2euler, quat_rot_vec, quat_conjugate, euler2quat

quaternion_invariants = np.concatenate(
    ((1.0/np.sqrt(2))*np.array([
        [1, 1, 0, 0],
        [1, -1, 0, 0],
        [1, 0, 1, 0],
        [1, 0, -1, 0],
        [1, 0, 0, 1],
        [1, 0, 0, -1]
    ]),
    np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, -1],
        [1, 0, 0, 0]
    ])), axis=0
)

class ReplayBuffer:
    def __init__(self, capacity, obs_dim, ac_dim, goal_dim, tau=4, per=False, e=0.01, a=0.6, image_obs=False,
                 filter_train_batch=False, random_future_goals=False, env_name="None"):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.ac_dim = ac_dim
        self.goal_dim = goal_dim
        self.tau = tau
        self.per = per
        self.top = 0
        self.curr_size = 0
        self.env_name = env_name
        self.filter_train_batch = filter_train_batch
        self.random_future_goals = random_future_goals
        if per:
            self.e = e
            self.a = a
            self.tree = np.zeros((2*capacity-1),)
        if image_obs:
            obs_dtype = np.uint8
        else:
            obs_dtype = np.float32
        self._observations = np.zeros((capacity, obs_dim), dtype=obs_dtype)
        self._next_observations = np.zeros((capacity, obs_dim), dtype=obs_dtype)  # easier to store repeats, despite inefficiency.
        self._achieved_goals = np.zeros((capacity, goal_dim), dtype=np.float32)
        self._actions = np.zeros((capacity, ac_dim), dtype=np.float32)
        self._idx_to_valid_future_goal = [None] * capacity

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
        obs = self._observations[data_indices, :] + noise[0] * np.random.randn(len(data_indices), self.obs_dim)
        action = np.clip(
            self._actions[data_indices, :] + noise[1] * np.random.randn(len(data_indices), self.ac_dim), -1.0, 1.0)
        next_obs = self._next_observations[data_indices, :] + noise[2] * np.random.randn(len(data_indices),
                                                                                         self.obs_dim)

        # if self.env_name == "fetch_push_ng":
        #     q_inv = quaternion_invariants[np.random.randint(0, len(quaternion_invariants), batch_size), :]
        #     q_c_obs = euler2quat(obs[..., 9:12])
        #     q_new_obs = quat_mul(q_c_obs, q_inv)
        #     euler_new_obs = quat2euler(q_new_obs)
        #     obs[..., 9:12] = np.sin(euler_new_obs)
        #     obs[..., 12:15] = np.cos(euler_new_obs)
        #     q_c_nobs = euler2quat(next_obs[..., 9:12])
        #     q_new_nobs = quat_mul(q_c_nobs, q_inv)
        #     euler_new_nobs = quat2euler(q_new_nobs)
        #     next_obs[..., 9:12] = np.sin(euler_new_nobs)
        #     next_obs[..., 12:15] = np.cos(euler_new_nobs)

        return obs, action, next_obs, tree_indices

    def sample_for_gan_training(self, batch_size, noise=[0.0, 0.0, 0.0]):
        traj_inds = np.zeros((batch_size, self.tau+1)).astype(int)
        goal_inds = np.zeros((batch_size, self.tau)).astype(int)
        c = 0
        if self.per:
            total = self._tree_total()
        while c < batch_size:
            if self.per:
                s = np.random.uniform(0, total)
                tree_idx, start_idx = self._get_indices(s)
            else:
                start_idx = int(np.random.randint(0, self.curr_size))
            if len(self._idx_to_valid_future_goal[start_idx]) < self.tau:
                continue
            else:
                possible_goals = self._idx_to_valid_future_goal[start_idx]
                possible_idxs = [np.random.randint(0, len(possible_goals))]
                if self.filter_train_batch:
                    start_goal = self._achieved_goals[start_idx, :]
                    end_goal = self._achieved_goals[possible_goals[possible_idxs[0]], :]
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

        # if self.env_name == "fetch_push_ng":
        #     #add random invariant rotation to cube - give examples where cube has been flipped.
        #     q_inv = quaternion_invariants[np.random.randint(0, len(quaternion_invariants), batch_size), :]
        #     q_inv = np.tile(q_inv[:, np.newaxis, :], (1, self.tau+1, 1))
        #     q_curr = euler2quat(obs[..., 9:12])
        #     q_new = quat_mul(q_curr.reshape(-1, 4), q_inv.reshape(-1,4)).reshape(q_inv.shape[0], q_inv.shape[1], 4)
        #     euler_new = quat2euler(q_new)
        #     obs[..., 9:12] = np.sin(euler_new)
        #     obs[..., 12:15] = np.cos(euler_new)

        return obs, actions, goals#, traj_inds, goal_inds

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

        # insert priorities
        if self.per:
            for idx in range(self.top, self.top + path_len):
                data_idx = idx % self.capacity
                tree_idx = data_idx + self.capacity - 1
                self._update(tree_idx, self._get_priority(errors[idx - self.top]))

        self.top = (self.top + path_len) % self.capacity
        self.curr_size = min(self.curr_size + path_len, self.capacity)

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

    def update_priorities(self, errors, tree_indices):
        for i in range(len(errors)):
            p = self._get_priority(errors[i])
            self._update(tree_indices[i], p)

if __name__ == "__main__":
    np.random.seed(10)
    buffer = ReplayBuffer(20, obs_dim=3, ac_dim=4, goal_dim=3, tau=4, per=True)

    path = {}
    path["observations"] = np.random.randn(12, 3)
    path["actions"] = np.random.randn(12, 4)
    path["next_observations"] = np.random.randn(12,3)
    path["achieved_goals"] = np.random.randn(12,3)
    path["errors"] = np.random.randn(12)

    buffer.add_path(path)

    path = {}
    path["observations"] = np.random.randn(12, 3)
    path["actions"] = np.random.randn(12, 4)
    path["next_observations"] = np.random.randn(12, 3)
    path["achieved_goals"] = np.random.randn(12, 3)
    path["errors"] = np.random.randn(12)

    buffer.add_path(path)

    obs_sample, acs_sample, goals_sample = buffer.sample_for_gan_training(10)

    print("OK?")
