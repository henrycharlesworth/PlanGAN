import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class PointEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    FILE = "point2.xml"

    def __init__(self, file_path=None, control_mode='linear'):
        self.control_mode = control_mode
        mujoco_env.MujocoEnv.__init__(self, file_path, 10)
        utils.EzPickle.__init__(self)

    def _step(self, a):
        return self.step(a)

    def _get_obs(self):
        pos = self.data.qpos.flat[:2]
        vel = self.data.qvel.flat[:2]
        if self.control_mode == 'pos':
            return pos
        else:
            return np.concatenate([pos, vel])

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        return ob, 0.0, False, {}

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 20
        self.viewer.cam.elevation = -90

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        return self._get_obs()
