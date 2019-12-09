import abc
import numpy as np
import torch

class Planner(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def generate_next_action(self, curr_state, target_goal, model, env):
        pass

    @abc.abstractmethod
    def generate_target_state(self, curr_goal, end_goal, GAN, env):
        pass


class StandardPlanner(Planner):
    def __init__(self, num_trajs_gan=1000, max_steps_gan=15, num_successes_to_terminate_gan=10,
                 num_trajs_model=1000, max_steps_model=7, device="cpu"):
        """
        num_trajs_gan: number of goal-conditioned coarse-grained trajectories we generate for "long-term" planning
        max_steps_gan: max number of steps (multiples of tau) that we plan with the GAN for
        num_successes_to_terminate_gan: terminate the search early if we find this many successful plans
        num_trajs_model: number of trajectories we use in the random shooting to get to the initial target state(s)
        max_steps_model: maximum number of random shooting steps
        """
        self.num_trajs_gan = num_trajs_gan
        self.max_steps_gan = max_steps_gan
        self.num_successes_to_terminate = num_successes_to_terminate_gan
        self.num_trajs_model = num_trajs_model
        self.max_steps_model = max_steps_model
        self.device = device

    def generate_target_state(self, curr_goal, end_goal, GAN, env, use_all_successes=False):
        """
        Conditional on the current state and final goal, we use the GAN to plan a coarse-grained trajectory
        to the final goal. Note that in general the curr_goal can be a different dimension from the end_goal -
        the end goal is a goal of the environment (e.g. position of object), the curr_goal is a subset of the
        current state and can include information about the object, robot pos etc.
        """
        prev_goals = np.tile(curr_goal, (self.num_trajs_gan, 1))
        end_goal_rep = np.tile(end_goal, (self.num_trajs_gan, 1))
        success = np.zeros((self.num_trajs_gan,)).astype(int)

        initial_next_states = GAN.generate(prev_goals, end_goal_rep) #next states tau steps away
        success += env.batch_goal_achieved(initial_next_states, end_goal_rep, multiple_target=False, final_goal=True)
        prev_goals = initial_next_states

        if np.sum(success) >= self.num_successes_to_terminate:
            pass
        else:
            step_count = 1
            while step_count < self.max_steps_gan:
                next_goals = GAN.generate(prev_goals, end_goal_rep)
                success += (1-success)*env.batch_goal_achieved(next_goals, end_goal_rep, multiple_target=False, final_goal=True)
                if np.sum(success) >= self.num_successes_to_terminate:
                    break
                else:
                    step_count += 1
                    prev_goals = next_goals
        if np.sum(success) == 0:
            return None, False
        else:
            success_inds = np.where(success)[0]
            candidate_states = initial_next_states[success_inds, :]
            if use_all_successes:
                return candidate_states, True
            else:
                COM = torch.mean(candidate_states, dim=0, keepdim=True)
                dist_from_COM = (candidate_states-COM).norm(dim=-1)
                return candidate_states[torch.argmin(dist_from_COM), :], True

    def generate_next_action(self, curr_state, target_goal, model, env, multiple_goals=False, final_goal=False):
        """
        Random shooting with learned model.
        """
        curr_states = np.tile(curr_state, (self.num_trajs_model, 1))
        init_actions = np.array([env.action_space.sample() for _ in range(self.num_trajs_model)])
        step_count = 0
        actions = init_actions
        while step_count < self.max_steps_model:
            delta_s = model.predict(curr_states, actions)
            curr_states = curr_states + delta_s
            actions = np.array([env.action_space.sample() for _ in range(self.num_trajs_model)])
            curr_goals = env.get_goal_from_state(curr_states, final_goal=final_goal)
            if multiple_goals:
                success, which_ind = env.batch_goal_achieved(curr_goals, target_goal, multiple_target=True,
                                                                 final_goal=final_goal)
                if np.sum(success) > 0:
                    inds = np.where(success)[0]
                    dists = np.linalg.norm(curr_goals[inds, :]-target_goal[which_ind[inds],:], axis=-1)
                    ind = inds[np.argmin(dists)]
                    return init_actions[ind, :], True
            else:
                success = env.batch_goal_achieved_goal(curr_goals, target_goal, multiple_target=False,
                                                       final_goal=final_goal)
                if np.sum(success) > 0:
                    inds = np.where(success)[0]
                    dists = np.linalg.norm(curr_goals[inds, :]-np.tile(target_goal, (len(inds), 1)), axis=-1)
                    ind = inds[np.argmin(dists)]
                    return init_actions[ind, :], True
        #if we can't get there, return a random action
        return env.action_space.sample(), False