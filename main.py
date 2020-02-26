import argparse
import os
import joblib
import torch
import numpy as np
import json

from envs import ENV_LIST, return_environment
from networks import OneStepModelFC, TrajGeneratorFC, DiscriminatorFC
from replay_buffer import ReplayBuffer
from one_step_model import SimpleOneStepModel
from imagination import ImaginationModule
from controller import Controller
from planner import TrajectoryFracPlanner, TrajectoryFracPlannerRotInvariant2

parser = argparse.ArgumentParser()
#main
parser.add_argument('--expt_name', type=str, default="default")
parser.add_argument('--env', type=str, default="fetch_reach_reduced")
parser.add_argument('--tau', type=int, default=5)
#data collection
parser.add_argument('--traj_len', type=int, default=50)
parser.add_argument('--init_rand_trajs', type=int, default=1000)
parser.add_argument('--filter_rand_trajs', dest='filter_rand_trajs', action='store_true')
parser.add_argument('--extra_trajs', type=int, default=5000)
parser.add_argument('--min_traj_len', type=int, default=20)
parser.add_argument('--exploration_noise', type=float, default=0.1)
parser.add_argument('--buffer_capacity', type=int, default=1000000)
parser.add_argument('--PER', dest='per', action='store_true')
parser.add_argument('--PER_e', type=float, default=0.01)
parser.add_argument('--PER_a', type=float, default=0.6)
#networks
parser.add_argument('--OSM_hidden_sizes', type=str, default="512,512")
parser.add_argument('--G_hidden_sizes', type=str, default="512,512")
parser.add_argument('--D_hidden_sizes', type=str, default="512,512")
parser.add_argument('--minmaxscaler', dest='minmaxscaler', action='store_true')
#GAN/one-step model training
parser.add_argument('--reg_gan_with_osm', dest='reg_gan_with_osm', action='store_true')
parser.add_argument('--gan_model_l2', type=float, default=0.3)
parser.add_argument('--use_osm_in_gan', dest='use_osm_in_gan', action='store_true')
parser.add_argument('--gen_next_z', dest='gen_next_z', action='store_true')
parser.add_argument('--use_same_z', dest='use_same_z', action='store_true')
parser.add_argument('--gan_latent_dim', type=int, default=64)
parser.add_argument('--DS', dest='div_sensitivity', action='store_true')
parser.add_argument('--DS_final_only', dest='ds_final_only', action='store_true')
parser.add_argument('--ds_coeff', type=float, default=1.0)
parser.add_argument('--l2_G', type=float, default=0.0001)
parser.add_argument('--l2_D', type=float, default=0.0001)
parser.add_argument('--l2_OSM', type=float, default=0.0001)
parser.add_argument('--G_optimiser', type=str, default="ADAM 0.0001 0.5 0.999")
parser.add_argument('--D_optimiser', type=str, default="ADAM 0.0001 0.5 0.999")
parser.add_argument('--OSM_optimiser', type=str, default="ADAM 0.001 0.9 0.999")
parser.add_argument('--init_gan_train_its', type=int, default=100000)
parser.add_argument('--init_osm_train_its', type=int, default=100000)
parser.add_argument('--train_per_extra_gan', type=int, default=250)
parser.add_argument('--train_per_extra_osm', type=int, default=250)
parser.add_argument('--gan_batch_size', type=int, default=128)
parser.add_argument('--osm_batch_size', type=int, default=256)
parser.add_argument('--gan_train_end', type=int, default=10000)
parser.add_argument('--osm_train_end', type=int, default=10000)
parser.add_argument('--filter_train_batch', dest='filter_train_batch', action='store_true')
parser.add_argument('--random_future_goals', dest='random_future_goals', action='store_true')
parser.add_argument('--num_gans', type=int, default=1)
parser.add_argument('--num_osms', type=int, default=1)
parser.add_argument('--use_all_osms_for_each_gan', dest='use_all_osms_for_each_gan', action='store_true')
#planner
parser.add_argument('--num_trajs_plan', type=int, default=1000)
parser.add_argument('--max_steps_plan', type=int, default=0) #0 means traj_len
parser.add_argument('--frac_best', type=float, default=0.05)
parser.add_argument('--num_reps_final', type=int, default=100)
parser.add_argument('--plan_alpha', type=float, default=1.0)
parser.add_argument('--planner_type', type=str, default="trajfrac") #DELETE THIS EVENTUALLY
parser.add_argument('--plan_average_ac', dest='plan_av_ac', action='store_true')

parser.add_argument('--no_extra_trajs', dest='run_extra_trajs', action='store_false')

parser.set_defaults(per=False, div_sensitivity=False, ds_final_only=False, gen_next_z=False,
                    use_osm_in_gan=False, reg_gan_with_osm=False, use_same_z=False, run_extra_trajs=True,
                    filter_rand_trajs=False, minmaxscaler=False, filter_train_batch=False,
                    random_future_goals=False, use_all_osms_for_each_gan=False, plan_av_ac=False)
args = parser.parse_args()
assert args.env in ENV_LIST
G_optim_opts = args.G_optimiser.split(" ")
D_optim_opts = args.D_optimiser.split(" ")
OSM_optim_opts = args.OSM_optimiser.split(" ")
G_hidden_sizes = np.array([int(item) for item in args.G_hidden_sizes.split(',')])
D_hidden_sizes = np.array([int(item) for item in args.D_hidden_sizes.split(',')])
OSM_hidden_sizes = np.array([int(item) for item in args.OSM_hidden_sizes.split(',')])
device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("experiments", exist_ok=True)
experiment_dir = "experiments/"+args.expt_name
os.makedirs(experiment_dir, exist_ok=True)
json.dump(vars(args), open(experiment_dir+"/experiment_args.json", "w"))

"""set up main components"""
tau = args.tau
env = return_environment(args.env)
state_dim = env.state_dim
ac_dim = env.ac_dim
goal_dim = env.goal_dim
#whether or not to train the one-step model.
if args.use_osm_in_gan or args.reg_gan_with_osm:
    train_osm = True
else:
    train_osm = False

OSM_nets = []
discrim_nets = []
generator_nets = []
for i in range(args.num_osms):
    OSM_nets.append(OneStepModelFC(state_dim, ac_dim, hidden_sizes=OSM_hidden_sizes, device=device).to(device))
for i in range(args.num_gans):
    discrim_nets.append(DiscriminatorFC(state_dim, goal_dim, ac_dim, hidden_sizes=D_hidden_sizes).to(device))
    generator_nets.append(TrajGeneratorFC(state_dim, ac_dim, goal_dim, args.gan_latent_dim, tau,
                       hidden_sizes=G_hidden_sizes, use_osm=args.use_osm_in_gan,
                       gen_next_z=args.gen_next_z, use_same_z=args.use_same_z).to(device))

OSM_opts = []
discrim_opts = []
generator_opts = []

for par, opts, nets in zip([G_optim_opts, D_optim_opts, OSM_optim_opts], [generator_opts, discrim_opts, OSM_opts],
                          [generator_nets, discrim_nets, OSM_nets]):
    if par[0] == "ADAM":
        lr = 0.001; betas = (0.9, 0.999); weight_decay = 0; #ADAM defaults
        if len(par) > 1:
            lr = float(par[1])
        if len(par) > 2:
            betas = (float(par[2]), float(par[3]))
        if len(par) > 4:
            weight_decay = float(par[4])
        for net in nets:
            opts.append(torch.optim.Adam(net.parameters(), lr=lr, betas=betas, weight_decay=weight_decay))
    elif par[0] == "SGD":
        lr = float(par[1]); momentum = 0; weight_decay = 0;
        if len(par) > 2:
            momentum = float(par[2])
        if len(par) > 3:
            weight_decay = float(par[3])
        for net in nets:
            opts.append(torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay))


OSM = SimpleOneStepModel(OSM_nets, OSM_opts, l2_reg=args.l2_OSM, device=device)
buffer = ReplayBuffer(capacity=args.buffer_capacity, obs_dim=state_dim, ac_dim=ac_dim, goal_dim=goal_dim, tau=tau,
                      per=args.per, e=args.PER_e, a=args.PER_a, filter_train_batch=args.filter_train_batch,
                      random_future_goals=args.random_future_goals, env_name=env.name)
imagination = ImaginationModule(generator_nets, discrim_nets, generator_opts, discrim_opts, OSM, args.gan_latent_dim, tau=tau,
                                div_sensitivity=args.div_sensitivity, ds_coeff=args.ds_coeff,
                                ds_final_only=args.ds_final_only, l2_reg_D=args.l2_D,
                                l2_reg_G=args.l2_G, reg_with_osm=args.reg_gan_with_osm,
                                l2_loss_coeff=args.gan_model_l2,
                                use_all_osms_for_each_gan= args.use_all_osms_for_each_gan, device=device)

planning_args = {"num_trajs": args.num_trajs_plan, "max_steps": args.max_steps_plan if args.max_steps_plan > 0 else args.traj_len,
                 "frac_best": args.frac_best, "num_reps_final": args.num_reps_final, "alpha": args.plan_alpha}

if args.planner_type == "trajfrac":
    planner = TrajectoryFracPlanner(planning_args)
elif args.planner_type == "trajfracRI":
    planning_args["return_average"] = args.plan_av_ac
    planner = TrajectoryFracPlannerRotInvariant2(planning_args)
else:
    raise ValueError("incorrect planner specified")

controller = Controller(env, imagination, buffer, planner, args.expt_name, init_rand_trajs=args.init_rand_trajs,
                        filter_rand_trajs = args.filter_rand_trajs,
                        extra_trajs=args.extra_trajs, traj_len=args.traj_len, min_traj_len=args.min_traj_len,
                        exploration_noise=args.exploration_noise, train_OSM=train_osm,
                        gan_batch_size=args.gan_batch_size, osm_batch_size=args.osm_batch_size,
                        init_train_gan=args.init_gan_train_its, init_train_OSM=args.init_osm_train_its,
                        gan_train_per_extra=args.train_per_extra_gan, osm_train_per_extra=args.train_per_extra_osm,
                        gan_train_end=args.gan_train_end, osm_train_end=args.osm_train_end)

controller.main_loop(init=True, extra_trajs=args.run_extra_trajs)