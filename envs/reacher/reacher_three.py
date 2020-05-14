import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import os

class ReacherThreeEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, threshold=0.02):
        self.threshold = threshold
        self.ac_dim = 3
        self.state_dim = 11
        self.goal_dim = 2
        self._max_episode_steps = 50
        self.goal = np.array([0.0,0.0])
        self.name = "reacher_three"
        self.object = False

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        mujoco_env.MujocoEnv.__init__(self, curr_dir+'/reacher_three.xml', 2)

    def _is_success(self, achieved_goal, goal):
        d = np.linalg.norm(achieved_goal-goal)
        return (d < self.threshold).astype(np.float32)

    def compute_reward(self, achieved_goal, goal, info):
        d = np.linalg.norm(achieved_goal-goal, axis=-1)
        return -(d > self.threshold).astype(np.float32)

    def save_state(self):
        return self.sim.get_state()

    def restore_state(self, state):
        self.sim.set_state(state)
        self.sim.forward()

    def step(self, a):
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        # reward_dist = - np.linalg.norm(vec)
        # reward_ctrl = - np.square(a).sum()
        # reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        obs = self._get_obs()
        reward = self.compute_reward(obs["achieved_goal"], self.goal, None)
        done = False
        info = {"is_success": self._is_success(obs["achieved_goal"], obs["desired_goal"])}
        return obs, reward, done, info

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        while True:
            self.goal = self.np_random.uniform(low=-.25, high=.25, size=2)
            if np.linalg.norm(self.goal) < 0.2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        obs = {}
        obs["observation"] = np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[:2],
            self.get_body_com("fingertip")[:2]
        ])
        obs["achieved_goal"] = self.get_body_com("fingertip")[:2]
        obs["desired_goal"] = self.goal

        return obs

    def get_goal_from_state(self, state):
        if len(state.shape) == 1:
            return state[-2:]
        else:
            return state[..., -2:]

    def batch_goal_achieved(self, states, goals, tol=0.02):
        states = self.get_goal_from_state(states)
        dists = np.linalg.norm(states-goals, axis=-1)
        return (dists < tol).astype(int)