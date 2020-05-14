import argparse
import os
import torch
import numpy as np
import json

torch.set_num_threads(1)

from envs import ENV_LIST, return_environment
from networks import OneStepModelFC, TrajGeneratorFC, DiscriminatorFC
from replay_buffer import ReplayBuffer
from one_step_model import SimpleOneStepModel
from imagination import ImaginationModule
from controller import Controller
from planner import IterativePlanner

parser = argparse.ArgumentParser()
#main
parser.add_argument('--expt_name', type=str, default="default")
parser.add_argument('--env', type=str, default="fetch_pick_and_place")
parser.add_argument('--tau', type=int, default=5)
#data collection
parser.add_argument('--traj_len', type=int, default=50)
parser.add_argument('--init_rand_trajs', type=int, default=250)
parser.add_argument('--filter_rand_trajs', dest='filter_rand_trajs', action='store_true')
parser.add_argument('--extra_trajs', type=int, default=3000)
parser.add_argument('--min_traj_len', type=int, default=20)
parser.add_argument('--exploration_noise', type=float, default=0.2)
parser.add_argument('--buffer_capacity', type=int, default=1000000)
#networks
parser.add_argument('--OSM_hidden_sizes', type=str, default="512,512")
parser.add_argument('--G_hidden_sizes', type=str, default="512,512")
parser.add_argument('--D_hidden_sizes', type=str, default="512,512")
#GAN/one-step model training
parser.add_argument('--reg_gan_with_osm', dest='reg_gan_with_osm', action='store_true')
parser.add_argument('--gan_model_l2', type=float, default=30.0)
parser.add_argument('--gan_latent_dim', type=int, default=64)
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
parser.add_argument('--big_trains_at', type=str, default="")
parser.add_argument('--filter_train_batch', dest='filter_train_batch', action='store_true')
parser.add_argument('--random_future_goals', dest='random_future_goals', action='store_true')
parser.add_argument('--num_gans', type=int, default=1)
parser.add_argument('--num_osms', type=int, default=1)
parser.add_argument('--use_all_osms_for_each_gan', dest='use_all_osms_for_each_gan', action='store_true')
#planner
parser.add_argument('--plan_num_init_acs', type=int, default=25)
parser.add_argument('--plan_alpha', type=float, default=5.0)
parser.add_argument('--plan_average_ac', dest='plan_av_ac', action='store_true')
parser.add_argument('--plan_num_copies', type=int, default=100)
parser.add_argument('--planner_osm_frac', type=float, default=0.5)
#just train on random trajs:
parser.add_argument('--no_extra_trajs', dest='run_extra_trajs', action='store_false')

parser.set_defaults(reg_gan_with_osm=True, run_extra_trajs=True,
                    filter_rand_trajs=False, filter_train_batch=False, random_future_goals=False,
                    use_all_osms_for_each_gan=False, plan_av_ac=False)
args = parser.parse_args()
assert args.env in ENV_LIST

big_trains = args.big_trains_at #stop generating new trajs and train for a while after these many extra trajectories are added.
if len(big_trains) > 0:
    big_trains = np.array([int(val) for val in big_trains.split(" ")])
else:
    big_trains = []

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
if args.reg_gan_with_osm:
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
                       hidden_sizes=G_hidden_sizes).to(device))

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
                      filter_train_batch=args.filter_train_batch, random_future_goals=args.random_future_goals, env_name=env.name)
imagination = ImaginationModule(generator_nets, discrim_nets, generator_opts, discrim_opts, OSM, args.gan_latent_dim, tau=tau,
                                l2_reg_D=args.l2_D, l2_reg_G=args.l2_G, reg_with_osm=args.reg_gan_with_osm,
                                l2_loss_coeff=args.gan_model_l2, use_all_osms_for_each_gan= args.use_all_osms_for_each_gan, device=device)

planning_args = {
    "num_acs": args.plan_num_init_acs, "max_steps": args.traj_len, "num_copies": args.plan_num_copies, "num_reps":1,
    "num_iterations":1, "alpha":args.plan_alpha, "osm_frac":args.planner_osm_frac, "return_average":args.plan_av_ac, "tol":0.05, "noise":0.2
}
planner = IterativePlanner(planning_args)

controller = Controller(env, imagination, buffer, planner, args.expt_name, init_rand_trajs=args.init_rand_trajs,
                        filter_rand_trajs = args.filter_rand_trajs,
                        extra_trajs=args.extra_trajs, traj_len=args.traj_len, min_traj_len=args.min_traj_len,
                        exploration_noise=args.exploration_noise, train_OSM=train_osm,
                        gan_batch_size=args.gan_batch_size, osm_batch_size=args.osm_batch_size,
                        init_train_gan=args.init_gan_train_its, init_train_OSM=args.init_osm_train_its,
                        gan_train_per_extra=args.train_per_extra_gan, osm_train_per_extra=args.train_per_extra_osm,
                        big_trains=big_trains)

controller.main_loop(init=True, extra_trajs=args.run_extra_trajs)