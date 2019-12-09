import abc
import numpy as np
import joblib
import os
from tensorboardX import SummaryWriter

import time

class Controller(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def generate_trajectory(self, add_to_buffer=True, render=False):
        pass

    @abc.abstractmethod
    def train_model(self, batch_size):
        pass

    @abc.abstractmethod
    def train_gan(self, batch_size):
        pass

    @abc.abstractmethod
    def load(self, buffer=True):
        pass

    @abc.abstractmethod
    def save(self, buffer=True):
        pass


class SimpleController(Controller):
    def __init__(self, env, planner, model, gan, replay_buffer, num_rand_trajs=200, num_add_trajs=300, traj_len=100,
                 gan_batch_size=64, model_batch_size=256, num_init_train_model=100000, num_init_train_gan=200000,
                 num_train_per_path_model=5000, num_train_per_path_gan=5000, num_eval_trajs=100,
                 gan_replan_every=-1, use_all_successes=False, min_traj_len=20, gan_sampling_noise=[0.0, 0.0, 0.0],
                 model_sampling_noise=[0.0, 0.0, 0.0], expt_name="default"):
        self.env = env
        self.planner = planner
        self.model = model
        self.gan = gan
        self.replay_buffer = replay_buffer
        self.expt_name = expt_name
        self.num_rand_trajs = num_rand_trajs
        self.num_add_trajs = num_add_trajs
        self.traj_len = traj_len
        self.gan_batch_size = gan_batch_size
        self.model_batch_size = model_batch_size
        self.num_init_train_model = num_init_train_model
        self.num_init_train_gan = num_init_train_gan
        self.num_train_per_path_model = num_train_per_path_model
        self.num_train_per_path_gan = num_train_per_path_gan
        self.num_eval_trajs = num_eval_trajs
        self.use_all_successes = use_all_successes
        self.gan_replan_every = gan.tau if gan_replan_every==-1 else gan_replan_every
        self.tau = gan.tau
        self.min_traj_len = min_traj_len
        self.gan_sampling_noise = gan_sampling_noise
        self.model_sampling_noise = model_sampling_noise
        self.train_steps_gan = 0
        self.train_steps_model = 0
        self.num_data_transitions = 0
        self.num_evaluations = 0
        self.print_every = 20
        self.writer = SummaryWriter("experiments/"+self.expt_name+"/logs")

    def main_loop(self, init=True, init_only=False):
        if init:
            t1 = time.time()
            #have to run initial random trajectories to populate the buffer
            for t in range(self.num_rand_trajs):
                self.generate_trajectory(rand_actions=True, verbose=True)
                self.num_data_transitions += self.traj_len
            t2 = time.time()
            t_traj = t2 - t1
            # fit scalers (only to initial data - could be a little problematic, potentially...)
            size = self.replay_buffer.curr_size
            states = self.replay_buffer._observations[:size, :]
            goals = self.replay_buffer._achieved_goals[:size, :]
            next_states = self.replay_buffer._next_observations[:size, :]
            states_m = states + self.model_sampling_noise[0]*np.random.randn(size, self.env.state_dim)
            next_states_m = next_states + self.model_sampling_noise[2]*np.random.randn(size, self.env.state_dim)
            state_diffs = next_states_m - states_m
            self.model.fit_scalers(states_m, state_diffs)
            states_cg = self.env.get_goal_from_state(states, final_goal=False)
            self.gan.fit_scaler(states_cg, goals)
            #run initial training on random data.
            t1 = time.time()
            for i in range(max(self.num_init_train_gan, self.num_init_train_model)):
                if i % self.print_every == 0:
                    verbose = True
                else:
                    verbose = False
                if self.train_steps_model < self.num_init_train_model:
                    self.train_model(self.model_batch_size, verbose=verbose)
                if self.train_steps_gan < self.num_init_train_gan:
                    self.train_gan(self.gan_batch_size, verbose=verbose)
            t2 = time.time()
            print("Time to generate trajs: %f seconds" % t_traj)
            print("Train time: %f seconds" % (t2-t1))
        self.full_eval()
        self.save(buffer=True)
        if init_only == False:
            num_data_transitions = int(self.num_data_transitions + self.traj_len*self.num_add_trajs)
            while self.num_data_transitions < num_data_transitions:
                traj_len = self.generate_trajectory(rand_actions=False)
                num_train_gan = int(self.num_train_per_path_gan*(traj_len/self.traj_len))
                num_train_model = int(self.num_train_per_path_model*(traj_len/self.traj_len))
                for i in range(max(num_train_gan, num_train_model)):
                    if i % 20 == self.print_every:
                        verbose = True
                    else:
                        verbose = False
                    if i < num_train_gan:
                        self.train_gan(self.gan_batch_size, verbose=verbose)
                    if i < num_train_model:
                        self.train_model(self.model_batch_size, verbose=verbose)
                    self.num_data_transitions += traj_len
                self.full_eval()

    def load(self, buffer=True):
        (model_dict, m_s_scaler, m_sd_scaler), (g_dict, d_dict, cg_scaler, goal_scaler) = \
            joblib.load("experiments/"+self.expt_name+"/parameters.pkl")
        self.model.load_params(model_dict, m_s_scaler, m_sd_scaler)
        self.gan.load_params(g_dict, d_dict, cg_scaler, goal_scaler)
        if buffer:
            self.replay_buffer = joblib.load("experiments/"+self.expt_name+"/replay_buffer.pkl")

    def save(self, buffer=True):
        model_params = self.model.get_params()
        gan_params = self.gan.get_params()
        joblib.dump((model_params, gan_params), "experiments/"+self.expt_name+"/parameters.pkl")
        if buffer:
            joblib.dump(self.replay_buffer, "experiments/"+self.expt_name+"/replay_buffer.pkl")

    def generate_trajectory(self, rand_actions=False, render=False, eval=False, verbose=False):
        path = {}
        obs = self.env.reset()
        curr_state = obs["observation"]
        curr_achieved_goal = obs["achieved_goal"]
        end_goal = obs["desired_goal"]
        curr_cg_state = self.env.get_goal_from_state(curr_state, final_goal=False) #can be different from state and achieved goal
        path["observations"] = np.zeros((self.traj_len, len(curr_state)))
        path["next_observations"] = np.zeros((self.traj_len, len(curr_state)))
        path["achieved_goals"] = np.zeros((self.traj_len, len(curr_achieved_goal)))
        path["actions"] = np.zeros((self.traj_len, self.model.action_dim))
        target_rec = []
        goal_achieved = False
        if rand_actions == False:
            target_state, plan_success = self.planner.generate_target_state(curr_cg_state, end_goal, self.gan, self.env,
                                                                            use_all_successes=self.use_all_successes)
            target_rec.append(target_state)
        else:
            target_state = None
        if render:
            self.env.render()
        curr_step = 0
        curr_step_this_target = 0
        while curr_step < self.traj_len:
            if rand_actions or target_state is None or goal_achieved:
                #if goal achieved we add random actions until we reach the minimum path length
                action = self.env.action_space.sample()
                final_goal = False
            else:
                success = self.env.batch_goal_achieved(target_state, end_goal, final_goal=True, multiple_target=False)
                if np.sum(success) > 0:
                    final_goal=True
                    target_state = end_goal
                else:
                    final_goal=False
                action, rs_succ = self.planner.generate_next_action(curr_state, target_state, multiple_target=self.use_all_successes,
                                                                    final_goal=final_goal)
            obs, _, _, _ = self.env.step(action)
            if render:
                self.env.render()
            next_state = obs["observation"]
            path["observations"][curr_step, :] = curr_state
            path["next_observations"][curr_step, :] = next_state
            path["actions"][curr_step, :] = action
            path["achieved_goals"][curr_step, :] = curr_achieved_goal
            curr_state = next_state
            curr_cg_state = self.env.get_goal_from_state(curr_state, final_goal=False)
            curr_achieved_goal = obs["achieved_goal"]
            curr_step += 1
            if self.env.batch_goal_achieved(curr_cg_state, end_goal, final_goal=True, multiple_target=False):
                if eval:
                    #we are just doing evaluation - path succeeded, so terminate.
                    return True, curr_step, path
                else:
                    goal_achieved = True
            if goal_achieved and curr_step >= self.min_traj_len and rand_actions==False:
                break
            #see if we need a new target
            if rand_actions == False:
                if self.use_all_successes:
                    target_success, _ = self.env.batch_goal_achieved(curr_cg_state, target_state, final_goal=False, multiple_target=True)
                else:
                    target_success = self.env.batch_goal_achieved(curr_cg_state, target_state, final_goal=False, multiple_target=False)
                if target_state is None or np.sum(target_success) > 0 or curr_step_this_target == self.gan_replan_every:
                    if rand_actions == False and final_goal == False:
                        target_state, plan_success = self.planner.generate_target_state(curr_cg_state, end_goal, self.gan,
                                                                                        self.env, use_all_successes=self.use_all_successes)
                        target_rec.append(target_state)
                        curr_step_this_target = 0
                else:
                    curr_step_this_target += 1
        path["observations"] = path["observations"][:curr_step, :]
        path["next_observations"] = path["next_observations"][:curr_step, :]
        path["actions"] = path["actions"][:curr_step, :]
        path["achieved_goals"] = path["achieved_goals"][:curr_step, :]
        if eval:
            if goal_achieved:
                return True, curr_step, path
            else:
                return False, _, path
        else:
            #add to buffer
            if self.replay_buffer.per:
                if rand_actions == False:
                    path["errors"] = self.model.get_errors(path["observations"], path["actions"],
                                                           path["next_observations"])
                else:
                    path["errors"] = np.ones((path["observations"].shape[0],))
            self.replay_buffer.add_path(path)
            if verbose:
                print("Trajectory of length %d generated. Random actions: %r. Buffer size: %d" %
                      (path["observations"].shape[0], rand_actions, self.replay_buffer.curr_size))
            return curr_step

    def train_model(self, batch_size, verbose=False):
        observations, actions, next_observations, tree_indices = \
            self.replay_buffer.sample_for_model_training(batch_size, noise=self.model_sampling_noise)
        m_loss, m_reg_loss = self.model.train_on_batch(observations, actions, next_observations)
        if self.replay_buffer.per:
            upd_errors = self.model.get_errors(observations, actions, next_observations)
            self.replay_buffer.update_priorities(upd_errors, tree_indices)
        self.train_steps_model += 1
        self.writer.add_scalar('loss_model_error', m_loss, self.train_steps_model)
        self.writer.add_scalar('loss_model_reg', m_reg_loss, self.train_steps_model)
        if verbose:
            print("Model training step %d. MSE loss: %f, regularisation loss: %f" % (self.train_steps_model, m_loss,
                                                                                     m_reg_loss))

    def train_gan(self, batch_size, verbose=False):
        observations, goals, target_observations = \
            self.replay_buffer.sample_for_gan(batch_size, noise=self.gan_sampling_noise)
        observations = self.env.get_goal_from_state(observations, final_goal=False)
        target_observations = self.env.get_goal_from_state(target_observations, final_goal=False)
        self.train_steps_gan += 1
        losses = self.gan.train_on_batch(observations, goals, target_observations)
        str_to_print = "GAN training step %d. " % self.train_steps_gan
        for k, v in losses.items():
            self.writer.add_scalar(k, v, self.train_steps_gan)
            str_to_print += "[%s: %f] " % (k, v)
        if verbose:
            print(str_to_print)

    def full_eval(self):
        """number of successes in 100 test trajs (w/ goals). number of steps for each success.
           gan avg distance to target - do object separately, perhaps. baseline can be random data.
        """
        self.num_evaluations += 1
        pass