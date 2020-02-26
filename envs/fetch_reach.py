from gym.envs.robotics import fetch_env, rotations
from gym import utils
from gym.envs.robotics.utils import robot_get_obs
import os
import numpy as np


class FetchReach(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, random_start=False, distance_threshold=0.05, possible_start_states=None, reduced=False):
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
        self.display_goal_marker = True
        self.possible_start_states = possible_start_states
        self.ac_dim = 4
        self.reduced = reduced
        self.object=False
        if reduced:
            self.name = "fetch_reach_reduced"
            self.state_dim = 3
        else:
            self.name = "fetch_reach"
            self.state_dim = 10
        self.goal_dim = 3
        fetch_env.FetchEnv.__init__(self, os.path.join('fetch', 'reach.xml'), has_object=False,
                                    block_gripper=True, n_substeps=self.n_substeps, gripper_extra_height=0.2,
                                    target_in_the_air=True, target_offset=0.0, obj_range=0.15, target_range=0.15,
                                    distance_threshold=distance_threshold, initial_qpos=initial_qpos,
                                    reward_type="sparse")
        utils.EzPickle.__init__(self)

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

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = robot_get_obs(self.sim)
        if self.has_object:
            object_pos = self.sim.data.get_site_xpos('object0')
            # rotations
            object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
            # velocities
            object_velp = self.sim.data.get_site_xvelp('object0') * dt
            object_velr = self.sim.data.get_site_xvelr('object0') * dt
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())
        obs = np.concatenate([
            grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        ])
        if self.reduced:
            obs = obs[:3]

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def batch_goal_achieved(self, states, goals, tol=0.05):
        states = self.get_goal_from_state(states)
        dists = np.linalg.norm(states-goals, axis=-1)
        return (dists < tol).astype(int)

    def get_goal_from_state(self, state):
        if len(state.shape) == 1:
            return state[:3]
        else:
            return state[..., :3]

    def get_robot_pos_from_state(self, states):
        if len(states.shape)==1:
            return states[:3]
        else:
            return states[:, :3]