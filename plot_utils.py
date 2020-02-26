import torch
import numpy as np
import json
import joblib
from argparse import Namespace
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from envs import ENV_LIST, return_environment
from networks import OneStepModelFC, TrajGeneratorFC, DiscriminatorFC
from replay_buffer import ReplayBuffer
from one_step_model import SimpleOneStepModel
from imagination import ImaginationModule
from controller import Controller

def load_experiment(expt_name, param_name=None, load_buffer=False, device="cuda", planner=None):
    path = "experiments/"+expt_name
    params = json.load(open(path+"/experiment_args.json", "r"))
    args = Namespace(**params)
    G_hidden_sizes = np.array([int(item) for item in args.G_hidden_sizes.split(',')])
    D_hidden_sizes = np.array([int(item) for item in args.D_hidden_sizes.split(',')])
    OSM_hidden_sizes = np.array([int(item) for item in args.OSM_hidden_sizes.split(',')])
    G_optim_opts = args.G_optimiser.split(" ")
    D_optim_opts = args.D_optimiser.split(" ")
    OSM_optim_opts = args.OSM_optimiser.split(" ")
    tau = args.tau
    env = return_environment(args.env)
    state_dim = env.state_dim
    ac_dim = env.ac_dim
    goal_dim = env.goal_dim
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
            lr = 0.001;
            betas = (0.9, 0.999);
            weight_decay = 0;  # ADAM defaults
            if len(par) > 1:
                lr = float(par[1])
            if len(par) > 2:
                betas = (float(par[2]), float(par[3]))
            if len(par) > 4:
                weight_decay = float(par[4])
            for net in nets:
                opts.append(torch.optim.Adam(net.parameters(), lr=lr, betas=betas, weight_decay=weight_decay))
        elif par[0] == "SGD":
            lr = float(par[1]);
            momentum = 0;
            weight_decay = 0;
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
    imagination = ImaginationModule(generator_nets, discrim_nets, generator_opts, discrim_opts, OSM,
                                    args.gan_latent_dim, tau=tau,
                                    div_sensitivity=args.div_sensitivity, ds_coeff=args.ds_coeff,
                                    ds_final_only=args.ds_final_only, l2_reg_D=args.l2_D,
                                    l2_reg_G=args.l2_G, reg_with_osm=args.reg_gan_with_osm,
                                    l2_loss_coeff=args.gan_model_l2,
                                    use_all_osms_for_each_gan=args.use_all_osms_for_each_gan, device=device)
    controller = Controller(env, imagination, buffer, planner, args.expt_name, init_rand_trajs=args.init_rand_trajs,
                            filter_rand_trajs=args.filter_rand_trajs,
                            extra_trajs=args.extra_trajs, traj_len=args.traj_len, min_traj_len=args.min_traj_len,
                            exploration_noise=args.exploration_noise, train_OSM=train_osm,
                            gan_batch_size=args.gan_batch_size, osm_batch_size=args.osm_batch_size,
                            init_train_gan=args.init_gan_train_its, init_train_OSM=args.init_osm_train_its,
                            gan_train_per_extra=args.train_per_extra_gan, osm_train_per_extra=args.train_per_extra_osm,
                            gan_train_end=args.gan_train_end, osm_train_end=args.osm_train_end)

    controller.load(buffer=load_buffer, name=param_name)
    return controller


def generate_trajectories(imagination, env, num_trajs, object=False, start_state=None, end_goal=None,
                          start_env_state=None, plot_exact=False, plot_model=True, end_points_only=False,
                          num_steps=None, return_diff=False, object_only=False, fixed_axes=False):
    """
    :param imagination: has the gan and OSM
    :param env:
    :param num_trajs: number of trajectories
    :param object: object and robot
    :param start_state: start position - will generate by resetting env if None
    :param end_goal:
    :param start_env_state: - if plot_exact and start_state provided then also need to provide this
    :param plot_exact: plot exact trajectory that generated actions produce
    :param plot_model: plot trajectory that OSM predicts from generated actions
    """
    gan_traj_properties = {"marker": "o", "markersize": 3, "markerfacecolor": (0,0,1,0.2), "markeredgecolor": (0,0,1,0.2),
                           "linestyle":"-", "color": (0,0,1,0.2)}
    model_traj_properties = {"marker": "s", "markersize": 3, "markerfacecolor": (1, 0, 0, 0.2),
                           "markeredgecolor": (1, 0, 0, 0.2), "linestyle": "-", "color": (1, 0, 0, 0.2)}
    emp_traj_properties = {"marker": ">", "markersize": 3, "markerfacecolor": (0, 0, 0, 0.05),
                           "markeredgecolor": (0, 0, 0, 0.2), "linestyle": "-", "color": (0, 0, 0, 0.2)}

    gan_traj_obj_properties = {"marker": "o", "markersize": 3, "markerfacecolor": (1, 0, 0, 0.2),
                           "markeredgecolor": (1, 0, 0, 0.2), "linestyle": "-", "color": (1, 0, 0, 0.2)}
    emp_traj_obj_properties = {"marker": "o", "markersize": 3, "markerfacecolor": (0, 1, 0, 0.2),
                               "markeredgecolor": (0, 1, 0, 0.2), "linestyle": "-", "color": (0, 1, 0, 0.2)}
    if num_steps is None:
        num_steps = imagination.tau
    if start_state is None:
        obs = env.reset()
        start_state = obs["observation"]
        if plot_exact:
            start_env_state = env.save_state()
        if end_goal is None:
            end_goal = obs["desired_goal"]
    tau = imagination.tau
    imagination.netG.eval()
    imagination.one_step_model.network.eval()
    vals = [] #to sort out axes at the end
    vals.append(start_state[0:3])
    vals.append(end_goal)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(start_state[0], start_state[1], start_state[2], s=60, c="red")
    ax.scatter(end_goal[0], end_goal[1], end_goal[2], s=60, c="black")
    if object:
        ax.scatter(start_state[3], start_state[4], start_state[5], s=60, c="white", edgecolors="red", linewidths=2)
        vals.append(start_state[3:6])
    gen_states, gen_actions, _ = imagination.test_trajectory(np.tile(start_state, (num_trajs, 1)), np.tile(end_goal, (num_trajs, 1)), num_steps=num_steps)
    for i in range(num_trajs):
        if object_only==False:
            if end_points_only:
                ax.scatter(gen_states[i, -1, 0], gen_states[i, -1, 1], gen_states[i, -1, 2], c=gan_traj_properties["color"])
            else:
                ax.plot(gen_states[i, :, 0], gen_states[i, :, 1], gen_states[i, :, 2], **gan_traj_properties)
            vals.append(gen_states[:, :, :3].reshape(-1, 3))
        if object:
            if end_points_only:
                ax.scatter(gen_states[i, -1, 3], gen_states[i, -1, 4], gen_states[i, -1, 5], c=gan_traj_obj_properties["color"])
            else:
                ax.plot(gen_states[i, :, 3], gen_states[i, :, 4], gen_states[i, :, 5], **gan_traj_obj_properties)
            vals.append(gen_states[:, :, 3:6].reshape(-1, 3))
    if plot_model:
        model_states = np.zeros_like(gen_states)
        model_states[:, 0, :] = gen_states[:, 0, :]
        for j in range(num_steps):
            model_states[:, j+1, :] = imagination.one_step_model.predict(model_states[:, j, :], gen_actions[:, j, :])
        vals.append(model_states[:, :, :3].reshape(-1, 3))
        if object:
            vals.append(model_states[:, :, 3:6].reshape(-1, 3))
        for i in range(num_trajs):
            if object_only == False:
                if end_points_only:
                    ax.scatter(model_states[i, -1, 0], model_states[i, -1, 1], model_states[i, -1, 2], c=model_traj_properties["color"])
                else:
                    ax.plot(model_states[i, :, 0], model_states[i, :, 1], model_states[i, :, 2], **model_traj_properties)
            if object:
                if end_points_only:
                    ax.scatter(model_states[i, -1, 3], model_states[i, -1, 4], model_states[i, -1, 5], c=model_traj_properties["color"])
                else:
                    ax.plot(model_states[i, :, 3], model_states[i, :, 4], model_states[i, :, 5], **model_traj_properties)
    if plot_exact:
        exact_states = np.zeros_like(gen_states)
        exact_states[:, 0, :] = gen_states[:, 0, :]
        for i in range(num_trajs):
            env.restore_state(start_env_state)
            for j in range(num_steps):
                obs, _, _, _ = env.step(gen_actions[i, j, :])
                exact_states[i, j+1, :] = obs["observation"][:gen_states.shape[-1]]
            if object_only == False:
                if end_points_only:
                    ax.scatter(exact_states[i, -1, 0], exact_states[i, -1, 1], exact_states[i, -1, 2], c=emp_traj_properties["color"])
                else:
                    ax.plot(exact_states[i, :, 0], exact_states[i, :, 1], exact_states[i, :, 2], **emp_traj_properties)
            if object:
                if end_points_only:
                    ax.plot(exact_states[i, -1, 3], exact_states[i, -1, 4], exact_states[i, -1, 5], c=emp_traj_obj_properties["color"])
                else:
                    ax.plot(exact_states[i, :, 3], exact_states[i, :, 4], exact_states[i, :, 5], **emp_traj_obj_properties)
        vals.append(exact_states[:, :, :3].reshape(-1, 3))
        if object:
            vals.append(exact_states[:, :, 3:6].reshape(-1, 3))
        if return_diff:
            return np.mean(np.linalg.norm(exact_states-gen_states, axis=-1))
    vals = np.vstack(vals)
    X = vals[:, 0]; Y = vals[:, 1]; Z = vals[:, 2]
    max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max() / 2.0
    mid_x = (X.max() + X.min()) * 0.5
    mid_y = (Y.max() + Y.min()) * 0.5
    mid_z = (Z.max() + Z.min()) * 0.5
    if fixed_axes:
        ax.set_xlim(0.8, 1.5)
        ax.set_ylim(0.8, 1.5)
        ax.set_zlim(0.0, 0.7)
    else:
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

if __name__ == "__main__":
    controller = load_experiment("FP_1000init_10kafter_OSMreg", param_name="final")

    generate_trajectories(controller.imagination, controller.env, 10, plot_exact=True, plot_model=False, num_steps=20,
                          object=True, object_only=False)
    #generate_trajectories(controller.imagination, controller.env, 20, plot_exact=True, plot_model=False, num_steps=5)
    #generate_trajectories(controller.imagination, controller.env, 1000, plot_exact=False, plot_model=False,
    #                      end_points_only=True, num_steps=5)

    print("OK")
