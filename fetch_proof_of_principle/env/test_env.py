import numpy as np
import time
from fetch_proof_of_principle.env.fetch_reach_mod import FetchReachMod

env = FetchReachMod()
env.display_goal_marker=False
env.reset()
env.render()

#TEST MARKERS / RESET
for _ in range(500):
    #test adding markers needed for visualising goals/next states.
    #env.add_markers([np.array([1.34, 0.749, 0.534]),
   #                  np.array([1.34, 0.949, 0.534])])
    #env.add_markers([env._get_obs()["achieved_goal"]])
    env.render()
    env.reset()


"""
for _ in range(1000):
    env.step(env.action_space.sample())
    #env.viewer._hide_overlay = True
    #env.viewer.vopt.geomgroup[0] ^= 1
    env.render()
    #test = env.render(mode="rgb_array")
"""