from gym.envs.robotics.fetch.pick_and_place import FetchPickAndPlaceEnv

env = FetchPickAndPlaceEnv()

t_len = 50
num_trajs = 100

for i in range(num_trajs):
    env.reset()
    env.render()
    for j in range(t_len):
        env.step(env.action_space.sample())
        env.render()