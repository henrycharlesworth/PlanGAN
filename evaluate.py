from plot_utils import load_experiment
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--expt_name', type=str, default="FetchPush", help='must match directory in experiments dir')
parser.add_argument('--num_trajectories', type=int, default=50, help='number of rollouts')
parser.add_argument('--no_render', dest='render', action='store_false', help='turn off rendering')
parser.set_defaults(render=True)
args = parser.parse_args()

expt_name = args.expt_name
num_trajectories = args.num_trajectories
render = args.render
controller, planner = load_experiment(expt_name, param_name="final", load_buffer=False)

num_successes = 0
t1 = time.time()
for i in range(num_trajectories):
    planner.reset()
    success, step = controller.generate_trajectory(eval=True, random=False, verbose=True,
                                                   render=render)
    num_successes += success
    print("Trajectory {}, success: {}".format(i, success))
t2 = time.time()
print("{}/{} successes! Took {} seconds".format(num_successes, num_trajectories, t2-t1))