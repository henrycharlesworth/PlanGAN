import argparse
import joblib
import os
import torch
import numpy as np
import json
import warnings
warnings.filterwarnings("ignore")
from full_algorithm.envs import ENV_LIST, return_environment
from full_algorithm.networks import DynamicsModel, Generator_FC, Discriminator_FC
from full_algorithm.replay_buffer import ReplayBuffer
from full_algorithm.environment_model import SimpleEnvironmentModel
from full_algorithm.imagination import ImaginationGAN
from full_algorithm.planner import StandardPlanner
from full_algorithm.controller import SimpleController

"""
Explanation of all of the parameters here.
"""

parser = argparse.ArgumentParser()
#main
parser.add_argument('--env', type=str, default="fetch_reach_rand")
parser.add_argument('--experiment_name', type=str, default="default_params")
#controller
parser.add_argument('--init_rand_trajs', type=int, default=10000)
parser.add_argument('--num_add_trajs', type=int, default=300)
parser.add_argument('--traj_len', type=int, default=100)
parser.add_argument('--min_traj_len', type=int, default=20)
parser.add_argument('--init_train_model', type=int, default=100000)
parser.add_argument('--init_train_gan', type=int, default=1000000)
parser.add_argument('--gan_batch_size', type=int, default=64)
parser.add_argument('--model_batch_size', type=int, default=256)
parser.add_argument('--per_path_train_model', type=int, default=2500)
parser.add_argument('--per_path_train_gan', type=int, default=5000)
parser.add_argument('--eval_trajs', type=int, default=100)
parser.add_argument('--gan_replan_every', type=int, default=-1)
parser.add_argument('--use_all_successes', dest='use_all_successes', action='store_true')
parser.add_argument('--gan_sampling_noise', type=str, default="0.0,0.0,0.0")
parser.add_argument('--model_sampling_noise', type=str, default="0.0,0.0,0.0")
#planner
parser.add_argument('--num_trajs_gan', type=int, default=1000)
parser.add_argument('--max_steps_gan', type=int, default=15)
parser.add_argument('--num_successes_to_terminate', type=int, default=10)
parser.add_argument('--num_trajs_model', type=int, default=1000)
parser.add_argument('--max_steps_model', type=int, default=7)
#GAN training
parser.add_argument('--tau', type=int, default=5)
parser.add_argument('--latent_dim', type=int, default=64)
parser.add_argument('--noSN', dest='spectral_norm', action='store_false')
parser.add_argument('--GP', dest='gradient_penalty', action='store_true')
parser.add_argument('--noDS', dest='diversity_sensitivity', action='store_false')
parser.add_argument('--gan_lr', type=float, default=0.0001)
parser.add_argument('--gan_betas', type=str, default="0.5,0.999")
parser.add_argument('--num_discrim_updates', type=int, default=1)
parser.add_argument('--gen_l2_reg', type=float, default=0.0001)
parser.add_argument('--dis_l2_reg', type=float, default=0.0001)
parser.add_argument('--ds_coeff', type=float, default=1.0)
parser.add_argument('--gp_coeff', type=float, default=10.0)
parser.add_argument('--no_batchnorm_gen', dest='gen_batch_norm', action='store_false')
#model training
parser.add_argument('--model_l2_reg', type=float, default=0.0001)
parser.add_argument('--model_lr', type=float, default=0.001)
parser.add_argument('--model_betas', type=str, default="0.9,0.999")
#networks
parser.add_argument('--model_hidden_sizes', type=str, default="512,512")
parser.add_argument('--gen_hidden_sizes', type=str, default="512,512")
parser.add_argument('--dis_hidden_sizes', type=str, default="512,512")
parser.add_argument('--noskipconns', dest='skip_conns', action='store_false')
#eplay buffer
parser.add_argument('--replay_capacity', type=int, default=100000)
parser.add_argument('--noPER', dest='per', action='store_false')
parser.add_argument('--per_e', type=float, default=0.01)
parser.add_argument('--per_a', type=float, default=0.6)

parser.set_defaults(use_all_successes=False, spectral_norm=True, gradient_penalty=False, diversity_sensitivity=True,
                    gen_batch_norm=True, skip_conns=True, per=False)
args = parser.parse_args()
assert args.env in ENV_LIST
gan_betas = np.array([float(item) for item in args.gan_betas.split(',')])
model_betas = np.array([float(item) for item in args.model_betas.split(',')])
gan_sampling_noise = np.array([float(item) for item in args.gan_sampling_noise.split(',')])
model_sampling_noise = np.array([float(item) for item in args.model_sampling_noise.split(',')])
model_hidden_sizes = np.array([int(item) for item in args.model_hidden_sizes.split(',')])
gen_hidden_sizes = np.array([int(item) for item in args.gen_hidden_sizes.split(',')])
dis_hidden_sizes = np.array([int(item) for item in args.dis_hidden_sizes.split(',')])

device = "cuda" if torch.cuda.is_available() else "cpu"
#device="cpu"
os.makedirs("experiments", exist_ok=True)
os.makedirs("experiments/"+args.experiment_name, exist_ok=True)

env = return_environment(args.env)
state_dim = env.state_dim
action_dim = env.ac_dim
state_cg_dim = env.state_cg_dim
goal_dim = env.goal_dim

model_net = DynamicsModel(state_dim=state_dim, action_dim=action_dim, hidden_sizes=model_hidden_sizes).to(device)
gen_net = Generator_FC(state_dim=state_cg_dim, goal_dim=goal_dim, latent_dim=args.latent_dim, out_dim=state_cg_dim,
                       hidden_sizes=gen_hidden_sizes, batch_norm=args.gen_batch_norm, skip_conns=args.skip_conns,
                       device=device).to(device)
dis_net = Discriminator_FC(state_dim=state_cg_dim, goal_dim=goal_dim, spectral_norm=args.spectral_norm,
                           skip_conns=args.skip_conns, hidden_sizes=dis_hidden_sizes, device=device).to(device)
buffer = ReplayBuffer(args.replay_capacity, state_dim, action_dim, goal_dim, args.tau, per=args.per, e=args.per_e,
                      a=args.per_a)
model = SimpleEnvironmentModel(model_net, l2_reg=args.model_l2_reg, lr=args.model_lr,
                               betas = model_betas, device=device)
gan = ImaginationGAN(gen_net, dis_net, grad_penalty=args.gradient_penalty,
                     diversity_sensitivity=args.diversity_sensitivity, lr=args.gan_lr, betas=gan_betas,
                     num_discrim_updates=args.num_discrim_updates, l2_reg_d=args.dis_l2_reg, l2_reg_g=args.gen_l2_reg,
                     ds_coeff=args.ds_coeff, gp_coeff=args.gp_coeff, tau=args.tau, device=device)
planner = StandardPlanner(num_trajs_gan=args.num_trajs_gan, max_steps_gan=args.max_steps_gan,
                          num_successes_to_terminate_gan=args.num_successes_to_terminate,
                          num_trajs_model=args.num_trajs_model, max_steps_model=args.max_steps_model, device=device)

controller = SimpleController(env, planner, model, gan, buffer, num_rand_trajs=args.init_rand_trajs,
                              num_add_trajs=args.num_add_trajs, traj_len=args.traj_len,
                              gan_batch_size=args.gan_batch_size, model_batch_size=args.model_batch_size,
                              num_init_train_model=args.init_train_model, num_init_train_gan=args.init_train_gan,
                              num_train_per_path_model=args.per_path_train_model, num_train_per_path_gan=args.per_path_train_gan,
                              num_eval_trajs=args.eval_trajs, gan_replan_every=args.gan_replan_every,
                              use_all_successes=args.use_all_successes, min_traj_len=args.min_traj_len,
                              gan_sampling_noise=gan_sampling_noise, model_sampling_noise=model_sampling_noise,
                              expt_name=args.experiment_name)

"""
dict_G, dict_D = joblib.load("../fetch_proof_of_principle/results/goal_cond/from_start_betterparams_1/parameters.pkl")
curr, goals, targets, z, z2, z_alt = joblib.load("test_batch.pkl")
gan.generator.load_state_dict(dict_G)
gan.discriminator.load_state_dict(dict_D)
controller.gan.train_on_batch(curr, goals, targets, z, z2, z_alt)
"""

"""TESTING:"""
#controller.generate_trajectory(rand_actions=True)
#controller.generate_trajectory(rand_actions=True)

#INIT
#get trajs
controller.main_loop(init_only=True)
"""
for i in range(1000):
    controller.generate_trajectory(rand_actions=True, verbose=True)
#fit scalers
size = controller.replay_buffer.curr_size
states = controller.replay_buffer._observations[:size, :]
goals = controller.replay_buffer._achieved_goals[:size, :]
next_states = controller.replay_buffer._next_observations[:size, :]
states_m = states + controller.model_sampling_noise[0]*np.random.randn(size, 10)
next_states_m = next_states + controller.model_sampling_noise[2]*np.random.randn(size, 10)
state_diffs = next_states_m - states_m
controller.model.fit_scalers(states_m, state_diffs)
states_cg = controller.env.get_goal_from_state(states, final_goal=False)
controller.gan.fit_scaler(states_cg, goals)

np.random.seed(10)
torch.manual_seed(10)

controller.save(buffer=True)
#controller.load(buffer=True)

observations = controller.gan.cg_state_scaler.transform(controller.replay_buffer._observations[:100000, :3])
observations = observations.reshape(1000,100,3)
s_ind = np.array(range(0, 100-5))
t_ind = np.array([i for i in range(1000) for _ in range(len(s_ind))])
s_ind = np.tile(s_ind, 1000)
states = observations[t_ind, s_ind, :]
f_states = observations[t_ind, s_ind+5, :]
dists = np.linalg.norm((states-f_states).reshape(-1, 3), axis=-1)
avg = np.mean(dists)
std = np.std(dists)

#controller.train_model(controller.model_batch_size, verbose=True)
controller.train_gan(10000, verbose=True)
"""

print("OK?")