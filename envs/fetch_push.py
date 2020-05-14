from gym.envs.robotics import fetch_env, rotations
from gym import utils
from gym.envs.robotics.utils import robot_get_obs
import os
import numpy as np

import joblib

default_initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
default_gripper_pos = np.array([1.35, 0.75, 0.414])

goal_range = [[1.20, 1.50], [0.60, 0.90]]

def in_goal_range(xy):
    if xy[0] > 1.20 and xy[0] < 1.50:
        if xy[1] > 0.60 and xy[1] < 0.90:
            return True
    return False

class FetchPush(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, distance_threshold=0.05, reduced=False, remove_gripper=False):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        self.ac_dim = 4
        self.reduced = reduced
        self.remove_gripper = remove_gripper
        self.object=True
        if reduced:
            self.name = "fetch_push_reduced"
            self.state_dim = 6
        elif remove_gripper:
            self.name = "fetch_push"
            self.state_dim = 24
        else:
            self.state_dim = 28
            self.name = "fetch_push"
        self.goal_dim = 3
        fetch_env.FetchEnv.__init__(self, os.path.join('fetch', 'push.xml'), has_object=True,
                                    block_gripper=True, n_substeps=20, gripper_extra_height=0.0,
                                    target_in_the_air=False, target_offset=0.0, obj_range=0.15, target_range=0.15,
                                    distance_threshold=distance_threshold, initial_qpos=initial_qpos, reward_type='sparse')
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
        if self.has_object:
            object_pos = self.sim.data.get_site_xpos('object0')
            # rotations
            object_rot_sin = rotations.mat2euler(self.sim.data.get_site_xmat('object0')) #MODIFIED - easier for model to not predict big changes.
            #object_rot_sin = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
            object_rot_cos = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
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
        if self.remove_gripper:
            obs = np.concatenate([
                grip_pos, object_pos.ravel(), object_rel_pos.ravel(), object_rot_sin.ravel(),
                object_rot_cos.ravel(),
                object_velp.ravel(), object_velr.ravel(), grip_velp,
            ])
        else:
            obs = np.concatenate([
                grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot_sin.ravel(), object_rot_cos.ravel(),
                object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
            ])
        if self.reduced:
            obs = obs[:6]

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

class FetchPushRandomStart(FetchPush):
    def _reset_sim(self):
        self.restore_state(saved_states[int(np.random.randint(0, len(saved_states)))])
        gripper_pos = self._get_obs()["observation"][:3]
        object_xpos = gripper_pos[:2]
        while np.linalg.norm(object_xpos-gripper_pos[:2]) < 0.1 or in_goal_range(object_xpos)==False:
            object_xpos = gripper_pos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
        object_qpos = self.sim.data.get_joint_qpos('object0:joint')
        assert object_qpos.shape == (7,)
        object_qpos[:2] = object_xpos
        self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        self.sim.forward()
        return True

    def _sample_goal(self):
        goal = default_gripper_pos + self.np_random.uniform(-self.target_range, self.target_range, size=3)
        goal += self.target_offset
        goal[2] = self.height_offset
        return goal.copy()
