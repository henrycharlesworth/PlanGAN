from gym.envs.robotics import fetch_env
from gym.envs.robotics import utils as robot_utils
from gym import utils
import os
import numpy as np

from full_algorithm.envs.envs import Environment

class FetchReach(Environment, fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, random_start=False, distance_threshold=0.05, possible_start_states=None):
        initial_qpos = {
            'robot0:slide0': 0.4049,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
        }
        self.n_substeps = 20
        self.random_start = random_start
        self.initial_gripper_xpos = np.array([1.34, 0.749, 0.534])
        self.goal_range = np.stack((self.initial_gripper_xpos - np.array([0.2, 0.2, 0.13]),
                                    self.initial_gripper_xpos + np.array([0.2, 0.2, 0.13])))
        fetch_env.FetchEnv.__init__(self, os.path.join('fetch', 'reach.xml'), has_object=False,
                                    block_gripper=True, n_substeps=self.n_substeps, gripper_extra_height=0.2,
                                    target_in_the_air=True, target_offset=0.0, obj_range=0.15, target_range=0.15,
                                    distance_threshold=distance_threshold, initial_qpos=initial_qpos,
                                    reward_type="sparse")
        utils.EzPickle.__init__(self)
        self.display_goal_marker = True
        self.possible_start_states = possible_start_states
        self.ac_dim = 4
        self.state_dim = 10
        self.state_cg_dim = 3
        self.goal_dim = 3
        self.final_goal_threshold = distance_threshold
        self.cg_goal_threshold = distance_threshold

    def save_state(self):
        return self.sim.get_state()

    def restore_state(self, state):
        self.sim.set_state(state)
        self.sim.forward()

    def _reset_sim(self):
        if self.random_start:
            idx = np.random.randint(0, len(self.possible_start_states))
            self.restore_state(self.possible_start_states[idx])
        else:
            self.restore_state(self.initial_state)
        self.sim.forward()
        return True

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        if self.display_goal_marker:
            self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        else:
            self.sim.model.site_pos[site_id] = np.array([10000000, 1000000, 1000000])
        self.sim.forward()
        # visualise markers
        """if self.viewer is not None:
            if self.markers is not None:
                for i in range(len(self.markers)):
                    self.viewer.add_marker(**self.markers[i])"""

    def _sample_goal(self):
        goal = np.random.uniform(self.goal_range[0], self.goal_range[1])
        return goal.copy()

    def batch_goal_achieved(self, current, target, multiple_target=False, final_goal=False):
        if final_goal:
            current = self.get_goal_from_state(current, final_goal=True)
            target = self.get_goal_from_state(target, final_goal=True)
        if multiple_target:
            n_curr = current.shape[0]
            n_tar = target.shape[0]
            target = np.tile(target, (n_curr, 1))
            current = np.repeat(current, n_tar, axis=0)
            dists = np.linalg.norm(current-target, axis=-1)
            if final_goal:
                success = dists < self.final_goal_threshold
            else:
                success = dists < self.cg_goal_threshold
            success = success.reshape(n_curr, n_tar)
            final_success = np.sum(success, axis=1)
            which_target = np.zeros((n_curr,)).astype(int)
            inds = np.where(final_success)[0]
            for ind in inds:
                which_target[ind] = np.random.choice(np.where(success[ind,:])[0])
            return final_success, which_target
        else:
            if len(current.shape)==1 or (len(current.shape)==2 and current.shape[0]==1):
                if len(target.shape)==1 or (len(target.shape)==2 and target.shape[0]==1):
                    dists = np.linalg.norm(current-target)
                else:
                    current = np.tile(current, (target.shape[0], 1))
                    dists = np.linalg.norm(current-target, axis=-1)
            else:
                if len(target.shape)==1 or (len(target.shape)==2 and target.shape[0]==1):
                    target = np.tile(target, (current.shape[0], 1))
                    dists = np.linalg.norm(current-target, axis=-1)
                else:
                    dists = np.linalg.norm(current-target, axis=-1)
            if final_goal:
                return dists < self.final_goal_threshold
            else:
                return dists < self.cg_goal_threshold

    def get_goal_from_state(self, state, final_goal=True):
        """More generally will have to check if state is full_state or cg_state"""
        if len(state.shape) == 1:
            return state[:3]
        else:
            return state[:, :3]

    def get_robot_pos_from_state(self, states):
        if len(states.shape)==1:
            return states[:3]
        else:
            return states[:, :3]