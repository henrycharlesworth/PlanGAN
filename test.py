from plot_utils import load_experiment
from planner import TrajectoryFracPlanner, TrajectoryFracPlannerRotInvariant, TrajectoryFracPlannerRotInvariant2, IterativePlanner, SuperBasicPlanner
import numpy as np
#planning_args = {"num_trajs": 1000, "max_steps": 50}
#planner = StupidlySimplePlanner(planning_args)

# planning_args = {"num_trajs": 1000, "max_steps": 50, "frac_best": 0.05, "num_reps_final": 100, "alpha": 10.0,
#                  "tol":0.05, "state_noise": 0.0, "return_average": False}

planning_args = {"num_acs": 25, "max_steps": 50, "num_copies": 100, "num_reps": 2, "num_iterations": 1, "alpha": 1.0,
                 "osm_frac": 0.5, "return_average": True, "tol": 0.05, "noise": 0.2}

# planning_args = {"num_acs": 5000, "noise": 0.0}

#planning_args = {"num_trajs": 1000, "max_steps": 50, "frac_best": 0.05, "num_reps_final": 100, "alpha": 1.0, "state_noise": 0.05}
#planner = SlightlyLessStupidPlanner(planning_args)
# planner = TrajectoryFracPlannerRotInvariant2(planning_args)
#planner = TrajectoryFracPlanner(planning_args)
planner = IterativePlanner(planning_args)

# planner = SuperBasicPlanner(planning_args)

#controller = load_experiment("FP_1000init_10kafter_OSMreg", param_name="final", planner=planner)
controller = load_experiment("FP_250init_2kafter_OSMreg_newplanner_fullstate_l2=30.0_smallbuffer_50k_explorationnoise=0.2_newsampling_randomfuturegoals_3GANS_3OSMsmean", param_name="final", planner=planner,
                             load_buffer=False)
#controller.exploration_noise = 0.3
num_successes = 0
num_trajs=200

steps = []

controller.traj_len = 75
for i in range(num_trajs):
    planner.prev_angle = np.array([0.0,0.0,0.0])
    success, step = controller.generate_trajectory(eval=True, random=False, verbose=True, render=False)
    num_successes += success
    if success:
        steps.append(step)
    print(i)
#controller.extra_trajs = 20
#controller.gan_train_per_extra = 10
#controller.osm_train_per_extra = 10
#controller.main_loop(init=False, extra_trajs=True)

print("%d / %d successes!" % (num_successes, num_trajs))
print("Average number of steps for success: %f" % np.mean(steps))