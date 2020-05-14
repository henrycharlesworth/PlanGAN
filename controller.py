import numpy as np
import joblib
import time
from tensorboardX import SummaryWriter

class Controller:
    """Generate trajectories, call to train, etc."""
    def __init__(self, env, imagination, replay_buffer, planner, expt_name, init_rand_trajs=1000,
                 filter_rand_trajs=False, extra_trajs=1000,
                 traj_len=50, min_traj_len=20, exploration_noise=0.1, train_OSM=True, gan_batch_size=64,
                 osm_batch_size=256, init_train_gan=100000, init_train_OSM=100000, gan_train_per_extra=250,
                 osm_train_per_extra=250, gan_sampling_noise=[0.0001,0.0001,0.0001],
                 osm_sampling_noise=[0.0001,0.0001,0.0001], big_trains=[], print_every=20, eval_every=25000, eval_trajs=200):
        self.expt_name = expt_name
        self.path_name = "experiments/"+expt_name
        self.env = env
        self.planner = planner
        self.imagination = imagination
        self.replay_buffer = replay_buffer
        self.init_rand_trajs = init_rand_trajs
        self.filter_rand_trajs = filter_rand_trajs
        self.extra_trajs = extra_trajs
        self.traj_len = traj_len
        self.min_traj_len = min_traj_len
        self.exploration_noise = exploration_noise
        self.train_OSM = train_OSM
        self.gan_batch_size = gan_batch_size
        self.osm_batch_size = osm_batch_size
        self.init_train_gan = init_train_gan
        self.init_train_OSM = init_train_OSM
        self.gan_train_per_extra = gan_train_per_extra
        self.osm_train_per_extra = osm_train_per_extra
        self.gan_sampling_noise = gan_sampling_noise
        self.osm_sampling_noise = osm_sampling_noise
        self.big_trains = big_trains
        self.print_every = print_every
        self.eval_every = eval_every
        self.full_eval_data = []
        self.eval_trajs = eval_trajs
        self.eval_num = 0
        self.prev_eval_at = 0
        self.train_steps_gan = 0
        self.train_steps_model = 0
        self.num_env_transitions = 0
        self.writer = SummaryWriter(self.path_name+"/logs")

    def main_loop(self, init=True, extra_trajs=False):
        if init:
            t1 = time.time()
            c = 0
            wasted_trajs = 0
            while c < self.init_rand_trajs:
                added, _ = self.generate_trajectory(eval=False, random=True, verbose=True)
                self.num_env_transitions += self.traj_len
                if added:
                    c += 1
                else:
                    wasted_trajs += 1
            t2 = time.time()
            t_rand_traj = t2-t1
            #fit scalers on random data.
            size = self.replay_buffer.curr_size
            states, _, next_states, _ = self.replay_buffer.sample_for_model_training(size)
            goals = self.replay_buffer._achieved_goals[:size, :]
            states_m = states + self.osm_sampling_noise[0]*np.random.randn(size, self.env.state_dim)
            next_states_m = next_states + self.osm_sampling_noise[1]*np.random.randn(size, self.env.state_dim)
            state_diffs = next_states_m - states_m

            if self.env.name.startswith("four_rooms"):
                #manually fit these
                rand_x = np.random.uniform(self.env.x_range[0], self.env.x_range[1], states_m.shape[0])
                rand_y = np.random.uniform(self.env.y_range[0], self.env.y_range[1], states_m.shape[0])
                states_m[:, 0] = rand_x
                states_m[:, 1] = rand_y
                goals[:, 0] = rand_x
                goals[:, 1] = rand_y

            self.imagination.one_step_model.fit_scalers(states_m, state_diffs, env=self.env)
            self.imagination.fit_scalers(goals)
            #train on random data
            t1 = time.time()
            for i in range(max(self.init_train_OSM, self.init_train_gan)):
                if i % self.print_every == 0:
                    verbose = True
                else:
                    verbose = False
                if self.train_steps_model < self.init_train_OSM:
                    self.train_model(self.osm_batch_size, verbose=verbose)
                if self.train_steps_gan < self.init_train_gan:
                    self.train_gan(self.gan_batch_size, verbose=verbose)
            t2 = time.time()
            t_train_rand = t2-t1
            print("Time taken to generate trajs: %f seconds" % t_rand_traj)
            print("Time taken to train on random trajs: %f seconds" % t_train_rand)
            self.save(buffer=True, name="after_rand")
            self.curr_env_transitions = self.num_env_transitions
            #self.full_eval()
        if extra_trajs:
            if len(self.big_trains)==0:
                big_trains_at = []
                big_train_ind = np.inf
            else:
                big_trains_at = (self.big_trains*self.traj_len + self.num_env_transitions).astype(int)
                big_train_ind = 0

            total_env_transitions = int(self.num_env_transitions + self.traj_len*self.extra_trajs)
            self.curr_env_transitions = self.num_env_transitions
            fail_adds = 0
            while self.curr_env_transitions < total_env_transitions:
                random = False
                if fail_adds > 2:
                    random = True
                added, traj_len = self.generate_trajectory(eval=False, random=random, verbose=True)
                if added == False:
                    fail_adds += 1
                    wasted_trajs += 1
                    self.num_env_transitions += traj_len
                    continue
                else:
                    fail_adds = 0
                self.num_env_transitions += traj_len
                self.curr_env_transitions += traj_len

                big_train = False
                if big_train_ind < len(big_trains_at):
                    if self.curr_env_transitions > big_trains_at[big_train_ind]:
                        big_train_ind += 1
                        big_train = True
                        for i in range(max(self.init_train_gan, self.init_train_OSM)):
                            if i % self.print_every == 0:
                                verbose = True
                            else:
                                verbose = False
                            if i < self.init_train_gan:
                                self.train_gan(self.gan_batch_size, verbose=verbose)
                            if i < self.init_train_OSM:
                                self.train_model(self.osm_batch_size, verbose=verbose)
                if big_train == False:
                    num_train_gan = int(self.gan_train_per_extra * (traj_len / self.traj_len))
                    num_train_osm = int(self.osm_train_per_extra * (traj_len / self.traj_len))
                    for i in range(max(num_train_gan, num_train_osm)):
                        if i % self.print_every == 0:
                            verbose = True
                        else:
                            verbose = False
                        if i < num_train_gan:
                            self.train_gan(self.gan_batch_size, verbose=verbose)
                        if i < num_train_osm:
                            self.train_model(self.osm_batch_size, verbose=verbose)

                if (self.curr_env_transitions - self.prev_eval_at) > self.eval_every:
                    self.full_eval()
                    self.save_full_eval()
                    self.save(buffer=False, name=str(self.eval_num))
            self.save(buffer=False, name="final")
            self.full_eval()
            self.save_full_eval()
            print("Number of wasted trajectories: %d" % wasted_trajs)

    def generate_trajectory(self, eval=False, random=True, render=False, verbose=False, return_path=False):
        """..."""
        for i in range(len(self.imagination.G_nets)):
            self.imagination.G_nets[i].eval()
        for i in range(len(self.imagination.one_step_model.networks)):
            self.imagination.one_step_model.networks[i].eval()

        path = {}
        obs = self.env.reset()
        self.planner.reset()
        curr_state = obs["observation"]
        curr_achieved_goal = obs["achieved_goal"]
        first_achieved_goal = obs["achieved_goal"]
        end_goal = obs["desired_goal"]
        path["observations"] = np.zeros((self.traj_len, len(curr_state)))
        path["next_observations"] = np.zeros((self.traj_len, len(curr_state)))
        path["achieved_goals"] = np.zeros((self.traj_len, len(curr_achieved_goal)))
        path["actions"] = np.zeros((self.traj_len, self.imagination.ac_dim))
        if return_path:
            path["start_state"] = self.env.save_state()
            path["end_goal"] = end_goal.copy()
        curr_step = 0
        planner_successes = 0
        goal_achieved = False
        if render:
            self.env.render()
        while curr_step < self.traj_len:
            if random or (goal_achieved and eval==False):
                action = self.env.action_space.sample()
            else:
                action, planner_success = self.planner.generate_next_action(curr_state, end_goal, self.imagination,
                                                                            self.env, **self.planner.planning_args)
                if eval == False:
                    action += self.exploration_noise*np.random.randn(self.env.ac_dim)
                    action = np.clip(action, -1.0, 1.0)
                planner_successes += planner_success
            obs, _, _, _ = self.env.step(action)
            if render:
                self.env.render()
            next_state = obs["observation"]
            path["observations"][curr_step, :] = curr_state
            path["next_observations"][curr_step, :] = next_state
            path["actions"][curr_step, :] = action
            path["achieved_goals"][curr_step, :] = curr_achieved_goal
            curr_state = next_state
            curr_achieved_goal = obs["achieved_goal"]
            curr_step += 1
            if self.env.name.startswith("fetch_slide"):
                if curr_step == self.traj_len:
                    if eval:
                        if self.env._is_success(curr_achieved_goal, end_goal):
                            if verbose:
                                print("Goal achieved at end of trajectory!")
                            return True, curr_step
                        else:
                            if verbose:
                                print("Failed to achieve goal!")
                            return False, _
            else:
                if self.env._is_success(curr_achieved_goal, end_goal):
                    if eval:
                        if verbose:
                            print("Goal achieved within %d steps!" % curr_step)
                        if return_path:
                            path["observations"] = path["observations"][:curr_step, :]
                            path["next_observations"] = path["next_observations"][:curr_step, :]
                            path["actions"] = path["actions"][:curr_step, :]
                            path["achieved_goals"] = path["achieved_goals"][:curr_step, :]
                            return True, curr_step, path
                        else:
                            return True, curr_step
                    else:
                        goal_achieved = True
            if goal_achieved and curr_step >= self.min_traj_len and random == False:
                break

        path["observations"] = path["observations"][:curr_step, :]
        path["next_observations"] = path["next_observations"][:curr_step, :]
        path["actions"] = path["actions"][:curr_step, :]
        path["achieved_goals"] = path["achieved_goals"][:curr_step, :]

        if eval:
            if verbose:
                print("Failed to achieve goal!")
            if return_path:
                return False, _, path
            else:
                return False, _

        added = False
        if self.filter_rand_trajs == False:
            self.replay_buffer.add_path(path)
            added = True
        else:
            final_achieved_goal = obs["achieved_goal"]
            if np.linalg.norm(first_achieved_goal - final_achieved_goal) > 0.05:
                if self.env.name.startswith("fetch_pick_and_place"):
                    if first_achieved_goal[2] - final_achieved_goal[2] > 0.1:
                        added = False #block has fallen off
                    else:
                        self.replay_buffer.add_path(path)
                        added = True
                else:
                    self.replay_buffer.add_path(path)
                    added = True

        if verbose and added:
            print("Trajectory of length %d generated. Random: %r. Buffer size: %d" % (
                path["observations"].shape[0], random, self.replay_buffer.curr_size
            ))
        if return_path:
            return added, curr_step, path
        else:
            return added, curr_step

    def train_model(self, batch_size, verbose=False):
        if self.train_OSM == False:
            return
        m_losses = []; m_reg_losses = []
        for i in range(len(self.imagination.one_step_model.networks)):
            observations, actions, next_observations, tree_indices = \
                self.replay_buffer.sample_for_model_training(batch_size, noise=self.osm_sampling_noise)
            m_loss, m_reg_loss = self.imagination.one_step_model.train_on_batch(observations, actions, next_observations, osm_ind=i)
            m_losses.append(m_loss)
            m_reg_losses.append(m_reg_loss)
            self.writer.add_scalar('loss_model_error_'+str(i), m_loss, self.train_steps_model)
            self.writer.add_scalar('loss_model_reg_'+str(i), m_reg_loss, self.train_steps_model)
        self.train_steps_model += 1
        if verbose:
            str_to_print = "Model training step %d."%self.train_steps_model
            for i in range(len(self.imagination.one_step_model.networks)):
                str_to_print += " MSE loss %d: %f. reg loss: %f." % (i, m_losses[i], m_reg_losses[i])
            str_to_print += " Mean MSE: %f" % np.mean(m_losses)
            print(str_to_print)

    def train_gan(self, batch_size, verbose=False):
        self.train_steps_gan += 1
        for i in range(len(self.imagination.G_nets)):
            self.imagination.G_nets[i].train()
            self.imagination.D_nets[i].train()
            observations, actions, goals = self.replay_buffer.sample_for_gan_training(batch_size, noise=self.gan_sampling_noise)
            losses = self.imagination.train_on_trajs(observations, goals, actions, object=self.env.object, gan_ind=i)
            str_to_print = "GAN_%d training step %d. " % (i, self.train_steps_gan)
            for k, v in losses.items():
                self.writer.add_scalar(k+"_"+str(i), v, self.train_steps_gan)
                str_to_print += "[%s: %f] " % (k, v)
            if verbose:
                print(str_to_print)

    def save_full_eval(self):
        file_name = self.path_name + "/eval.pkl"
        joblib.dump(self.full_eval_data, file_name)

    def save(self, buffer=False, name=None):
        file_name = self.path_name + "/parameters"
        if name is not None:
            file_name += "_"+name
        file_name += ".pkl"
        params = self.imagination.save(one_step_also=True)
        joblib.dump(params, file_name)
        if buffer:
            joblib.dump(self.replay_buffer, self.path_name+"/replay_buffer.pkl")

    def load(self, buffer=False, name=None):
        file_name = self.path_name + "/parameters"
        if name is not None:
            file_name += "_" + name
        file_name += ".pkl"
        gan_params, osm_params = joblib.load(file_name)
        self.imagination.load(gan_params[0], gan_params[1], gan_params[2], osm_params)
        if buffer:
            self.replay_buffer = joblib.load(self.path_name +"/replay_buffer.pkl")

    def full_eval(self):
        orig_traj_len = self.traj_len
        if self.env.name.startswith("fetch") or self.env.name.startswith("four_rooms") or self.env.name.startswith("reacher"):
            self.traj_len = 50
        eval_data = {"num_env_transitions": self.num_env_transitions, "training_steps": [self.train_steps_gan, self.train_steps_model],
                     "n": self.eval_num, "buffer_size": self.replay_buffer.curr_size, "num_successes": 0, "num_steps": []}
        self.prev_eval_at = self.curr_env_transitions

        for i in range(self.eval_trajs):
            success, num_steps = self.generate_trajectory(eval=True, random=False, verbose=False)
            if success:
                eval_data["num_successes"] += 1
                eval_data["num_steps"].append(num_steps)
        eval_data["frac_success"] = eval_data["num_successes"] / self.eval_trajs
        self.full_eval_data.append(eval_data)
        self.eval_num += 1
        self.traj_len = orig_traj_len