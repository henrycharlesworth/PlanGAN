from gym.envs.robotics import fetch_env, rotations
from gym import utils
import os
from gym.envs.robotics.utils import robot_get_obs
import numpy as np

class FetchPickAndPlace(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, distance_threshold=0.05):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        self.ac_dim = 4
        self.goal_dim = 3
        self.object = True
        self.state_dim = 28
        self.name = "fetch_pick_and_place"
        fetch_env.FetchEnv.__init__(
            self, os.path.join('fetch', 'pick_and_place.xml'), has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=distance_threshold,
            initial_qpos=initial_qpos, reward_type='sparse')
        utils.EzPickle.__init__(self)

    def save_state(self):
        return self.sim.get_state()

    def restore_state(self, state):
        self.sim.set_state(state)
        self.sim.forward()

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = robot_get_obs(self.sim)
        object_pos = self.sim.data.get_site_xpos('object0')
        # rotations
        object_rot_sin = np.sin(rotations.mat2euler(self.sim.data.get_site_xmat('object0'))) #MODIFIED - easier for model to not predict big changes.
        #object_rot_sin = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
        object_rot_cos = np.cos(rotations.mat2euler(self.sim.data.get_site_xmat('object0')))
        # velocities
        object_velp = self.sim.data.get_site_xvelp('object0') * dt
        object_velr = self.sim.data.get_site_xvelr('object0') * dt
        # gripper state
        object_rel_pos = object_pos - grip_pos
        object_velp -= grip_velp

        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())

        obs = np.concatenate([
            grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot_sin.ravel(), object_rot_cos.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        ])
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
            return state[3:6]
        else:
            return state[..., 3:6]

    def get_robot_pos_from_state(self, states):
        if len(states.shape)==1:
            return states[:3]
        else:
            return states[..., :3]

    def get_obj_pos_from_state(self, states):
        if len(states.shape)==1:
            return states[3:6]
        else:
            return states[..., 3:6]