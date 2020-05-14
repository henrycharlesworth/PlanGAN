import numpy as np
from envs.four_rooms.create_maze_env import create_maze_env

def get_goal_sample_fn(env_name):
    if env_name == 'AntMaze':
        # NOTE: When evaluating (i.e. the metrics shown in the paper,
        # we use the commented out goal sampling function.    The uncommented
        # one is only used for training.
        #return lambda: np.array([0., 16.])
        rectangle_1 = np.array([[-1.41, -1.85], [1.40, 0.87]])
        rectangle_2 = np.array([[2.59, -1.85], [5.42, 0.87]])
        rectangle_3 = np.array([[-1.41, -5.86], [1.40, -3.17]])
        rectangle_4 = np.array([[2.59, -5.86], [5.42, -3.17]])
        rectangles = [rectangle_1, rectangle_2, rectangle_3, rectangle_4]
        areas = []
        for rectangle in rectangles:
            areas.append(np.abs(rectangle[0][0]-rectangle[1][0])*np.abs(rectangle[0][1]-rectangle[1][1]))
        sum_areas = np.sum(areas)
        areas = np.array(areas) / sum_areas
        areas = np.cumsum(areas)

        def sample_goal():
            r = np.random.rand()
            if r < areas[0]:
                ind = 0
            elif r < areas[1]:
                ind = 1
            elif r < areas[2]:
                ind = 2
            else:
                ind = 3
            x = np.squeeze(np.random.uniform(rectangles[ind][0][0], rectangles[ind][1][0], 1))
            y = np.squeeze(np.random.uniform(rectangles[ind][0][1], rectangles[ind][1][1], 1))
            return np.array([x,y])

        return sample_goal
    else:
        assert False, 'Unknown env'



class EnvWithGoal(object):
    def __init__(self, base_env, env_name, max_steps=50):
        self.base_env = base_env
        self.goal_sample_fn = get_goal_sample_fn(env_name)
        self.goal = None
        self.distance_threshold = 0.1
        self._max_episode_steps = max_steps
        import copy
        self.action_space = copy.deepcopy(self.base_env.action_space)
        self.action_space.high = np.ones((2,))
        self.action_space.low = -1*np.ones((2,))
        self.name = "four_rooms"
        self.object=False
        self.ac_dim = 2
        self.state_dim = 4
        self.goal_dim = 2
        self.x_range = [-1.41, 5.42]
        self.y_range = [-5.86, 0.91]

    def seed(self, num):
        return

    def reset(self):
        # self.viewer_setup()
        obs = self.base_env.reset()
        self.goal = self.goal_sample_fn()

        return {"observation": obs, "achieved_goal": obs[:2], "desired_goal": self.goal.copy()}

    def _is_success(self, achieved_goal, desired_goal):
        d = np.linalg.norm(achieved_goal - desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def compute_reward(self, achieved_goal, goal, info):
        d = np.linalg.norm(achieved_goal-goal, axis=-1)
        return -(d > self.distance_threshold).astype(np.float32)

    def step(self, a):
        obs, _, done, info = self.base_env.step(a)
        obs = {"observation": obs, "achieved_goal": obs[:2], "desired_goal": self.goal.copy()}
        info = {
            "is_success": self._is_success(obs["achieved_goal"], obs["desired_goal"])
        }
        reward = self.compute_reward(obs["achieved_goal"], self.goal, info)
        return obs, reward, done, info

    def batch_goal_achieved(self, states, goals, tol=0.1):
        dists = np.linalg.norm(states[..., :2]-goals, axis=-1)
        return (dists < tol).astype(int)

    def render(self):
        self.base_env.render()

    def save_state(self):
        return self.base_env.wrapped_env.sim.get_state()

    def restore_state(self, state):
        self.base_env.wrapped_env.sim.set_state(state)
        self.base_env.wrapped_env.sim.forward()

    # @property
    # def action_space(self):
    #     space = self.base_env.action_space
    #     space.high *= (1/30.0)
    #     space.low *= (1/30.0)
    #     return space


def return_standard_four_rooms():
    return EnvWithGoal(
        create_maze_env("AntMaze"),
        "AntMaze"
    )