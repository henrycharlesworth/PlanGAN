from gym.envs.robotics import fetch_env
from gym.envs.robotics import utils as robot_utils
from gym import utils
import os
import numpy as np
import joblib

MODEL_XML_PATH = os.path.join('fetch', 'reach.xml')

class FetchReachMod(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', n_substeps=20, distance_threshold=0.05, random_start=False,
                 possible_start_states=None):
        initial_qpos = {
            'robot0:slide0': 0.4049,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
        }
        self.n_substeps = n_substeps
        self.random_start = random_start
        self.initial_gripper_xpos = np.array([1.34, 0.749, 0.534])
        self.goal_range = np.stack((self.initial_gripper_xpos - np.array([0.2, 0.2, 0.13]),
                                    self.initial_gripper_xpos + np.array([0.2, 0.2, 0.13])))
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=False, block_gripper=True, n_substeps=n_substeps,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=distance_threshold,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)
        self.markers = []
        self.marker_type = 2
        self.marker_rgba = np.array([0.0, 0.0, 1.0, 1.0])
        self.marker_size = np.ones(3)*0.02
        self.display_goal_marker = True
        self.possible_start_states = possible_start_states

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

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:gripper_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 1.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def change_camera(self, ind):
        self.viewer.cam.fixedcamid = ind
        self.viewer.cam.type = 2#const.CAMERA_FIXED

    def _sample_goal(self):
        goal = np.random.uniform(self.goal_range[0], self.goal_range[1])
        return goal.copy()

    def reset_markers(self):
        self.markers = []

    def add_markers(self, markers, use_default_settings=True):
        if use_default_settings == False:
            for marker in markers:
                self.markers.append(marker)
        else:
            for marker in markers:
                self.markers.append({"pos": marker, "rgba": self.marker_rgba, "type": self.marker_type,
                                     "size": self.marker_size})

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        if self.display_goal_marker:
            self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        else:
            self.sim.model.site_pos[site_id] = np.array([10000000,1000000,1000000])
        self.sim.forward()
        #visualise markers
        if self.viewer is not None:
            if self.markers is not None:
                for i in range(len(self.markers)):
                    self.viewer.add_marker(**self.markers[i])

    def within_goal_range(self, achieved_goal):
        if achieved_goal[0] > self.goal_range[0][0] and achieved_goal[0] < self.goal_range[1][0] and \
                achieved_goal[1] > self.goal_range[0][1] and achieved_goal[1] < self.goal_range[1][1] and \
                achieved_goal[2] > self.goal_range[0][2] and achieved_goal[2] < self.goal_range[1][2]:
            return True
        else:
            return False



"""
def _env_setup(self, initial_qpos):
        #need to overwrite this to make starting position consistent for all n_substeps
        #(exact for factors of 20).
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        robot_utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos(
            'robot0:grip')
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(int(200/self.n_substeps)):
            self.sim.step()
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        
NOTE: can't remember why I needed this, but it was in PycharmProjects/planning_FetchEnvs/envs/fetch/fetch_reach_mod_random_start_pos.py
if we change num substeps then I think this might be necessary...
"""
