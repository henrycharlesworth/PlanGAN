import numpy as np


from gym.envs.robotics.rotations import quat_mul, euler2quat, quat2euler, mat2euler, quat_rot_vec, quat_conjugate

"""
other planner ideas:
1. simple but iteratively refine - at each iteration, keep the best 10% (or just successful ones, whichever is more).
Clone each of these, but at a random point in each of the clones (before achieving the goal) re-run the imagination GAN.
Pick the best trajectory at the end.

2. weighting trajectory w_i = e^(-alpha*numsteps_i/tau) * success_i. Could weight each initial action with this -
potentially very noisy. Could clone each initial action and run N extra trajs forward - whichever initial action (or could
do a mini trajectory of initial actions) leads to the highest average score.
"""


# class SlightlyLessStupidPlanner:
#     def __init__(self, planning_args = {}):
#         self.planning_args = planning_args
#
#     def generate_next_action(self, curr_state, end_goal, imagination, env, num_trajs=1000, max_steps=50,
#                              frac_best=0.1, num_reps_final=250, alpha=1.0):
#         gen_states, gen_actions, _ = imagination.test_trajectory(np.tile(curr_state, (num_trajs,1)), np.tile(end_goal, (num_trajs,1)), num_steps=max_steps)
#         success_mat = env.batch_goal_achieved(gen_states, end_goal[np.newaxis, np.newaxis, :])
#         success_inds = np.argmax(success_mat, axis=-1)
#         if np.max(success_inds) == 0:
#             return env.action_space.sample(), 0
#         valid_idx = np.where(success_inds > 0)[0]
#         best_inds = valid_idx[success_inds[valid_idx].argsort()]
#         num_best = int(np.floor(frac_best*num_trajs))
#         if len(best_inds) > num_best:
#             best_inds = best_inds[:num_best]
#         success_inds_best = success_inds[best_inds]
#         num_init = len(best_inds)
#         init_actions = gen_actions[best_inds, 0, :]
#         start_states = np.repeat(gen_states[best_inds, 1, :], num_reps_final, axis=0)
#         end_goals = np.tile(end_goal, (start_states.shape[0], 1))
#         gen_states_2, _, _ = imagination.test_trajectory(start_states, end_goals, num_steps=max_steps-1)
#         success_mat = env.batch_goal_achieved(gen_states_2, end_goals[:, np.newaxis, :])
#         success_inds = np.argmax(success_mat, axis=-1).reshape(num_init, num_reps_final)
#         success_inds = np.concatenate((success_inds, success_inds_best[:, np.newaxis]), axis=1)
#         scores = np.sum((success_inds > 0)*np.exp(-alpha*(success_inds - np.min(success_inds[success_inds>0]))), axis=-1)
#         return init_actions[np.argmax(scores), :], 1


class TrajectoryFracPlanner:
    def __init__(self, planning_args={}):
        self.planning_args = planning_args

    def generate_next_action(self, curr_state, end_goal, imagination, env, num_trajs=1000, max_steps=50,
                             frac_best=0.05, num_reps_final=250, alpha=1.0, state_noise=0.0, tol=0.05):
        num_gans = len(imagination.G_nets)
        num_trajs_per_gan = num_trajs // num_gans
        gen_states, gen_actions, _ = imagination.test_trajectory(np.tile(curr_state, (num_trajs_per_gan, 1)),
                                                                 np.tile(end_goal, (num_trajs_per_gan, 1)), num_steps=max_steps,
                                                                 state_noise=state_noise, gan_ind=0)
        for i in range(1, num_gans):
            g_states, g_actions, _ = imagination.test_trajectory(np.tile(curr_state, (num_trajs_per_gan, 1)),
                                                                 np.tile(end_goal, (num_trajs_per_gan, 1)), num_steps=max_steps,
                                                                 state_noise=state_noise, gan_ind=i)
            gen_states = np.concatenate((gen_states, g_states), axis=0)
            gen_actions = np.concatenate((gen_actions, g_actions), axis=0)

        success_mat = env.batch_goal_achieved(gen_states, end_goal[np.newaxis, np.newaxis, :], tol=tol)
        success_fraction = np.sum(success_mat, axis=-1) / max_steps
        success_inds = np.where(success_fraction>0)[0]
        if len(success_inds) == 0:
            return env.action_space.sample(), 0
        num_best = int(np.floor(frac_best*num_trajs))
        if len(success_inds) < num_best:
            num_best = len(success_inds)
        best_inds = (-success_fraction).argsort()[:num_best]
        success_frac_best = success_fraction[best_inds]

        num_reps_final_per_gan = num_reps_final // num_gans
        init_actions = gen_actions[best_inds, 0, :]
        start_states = np.repeat(gen_states[best_inds, 1, :], num_reps_final_per_gan, axis=0)
        end_goals = np.tile(end_goal, (start_states.shape[0], 1))

        scores = []
        for i in range(num_gans):
            gen_states_2, _, _ = imagination.test_trajectory(start_states, end_goals, num_steps=max_steps-1,
                                                             state_noise=state_noise, gan_ind=i)
            success_mat = env.batch_goal_achieved(gen_states_2, end_goals[:, np.newaxis, :], tol=tol)
            success_fraction_2 = np.sum(success_mat, axis=-1).reshape(num_best, num_reps_final_per_gan)
            success_fraction = np.concatenate((success_fraction_2, success_frac_best[:, np.newaxis]), axis=1)
            scores.append(np.sum(np.exp(alpha * success_fraction), axis=-1))

        scores = np.stack(scores)
        scores = np.mean(scores, axis=0)

        return init_actions[np.argmax(scores), :], 1

class TrajectoryFracPlannerRotInvariant:
    def __init__(self, planning_args={}):
        self.planning_args = planning_args
        self.quaternion_invariants = np.concatenate(
            ((1.0 / np.sqrt(2)) * np.array([
                [1, 1, 0, 0],
                [1, -1, 0, 0],
                [1, 0, 1, 0],
                [1, 0, -1, 0],
                [1, 0, 0, 1],
                [1, 0, 0, -1]
            ]),
             np.array([
                 [0, 1, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1],
                 [0, -1, 0, 0],
                 [0, 0, -1, 0],
                 [0, 0, 0, -1],
                 [1, 0, 0, 0]
             ])), axis=0
        )  # invariants of cube.
        self.prev_angle = np.array([0.0, 0.0, 0.0])

    def generate_next_action(self, curr_state, end_goal, imagination, env, num_trajs=1000, max_steps=50,
                             frac_best=0.05, num_reps_final=250, alpha=1.0, state_noise=0.0, tol=0.05):
        state = env.save_state()
        quaternions = np.tile(state.qpos[-4:], (len(self.quaternion_invariants), 1))
        quaternion_rot = quat_mul(quaternions, self.quaternion_invariants)
        euler_angles = quat2euler(quaternion_rot)
        scores = np.linalg.norm(euler_angles[:, :2], axis=-1) + np.abs(
            euler_angles[:, -1] - np.tile(self.prev_angle[-1], len(euler_angles)))
        ind = np.argmin(scores)
        self.prev_angle = euler_angles[ind, :]
        state.qpos[-4:] = quaternion_rot[ind, :]
        state.qvel[-3:] = quat_rot_vec(quat_conjugate(self.quaternion_invariants[ind]), state.qvel[-3:])
        env.restore_state(state)
        curr_state = env._get_obs()["observation"]

        num_gans = len(imagination.G_nets)
        num_trajs_per_gan = num_trajs // num_gans
        gen_states, gen_actions, _ = imagination.test_trajectory(np.tile(curr_state, (num_trajs_per_gan, 1)),
                                                                 np.tile(end_goal, (num_trajs_per_gan, 1)), num_steps=max_steps,
                                                                 state_noise=state_noise, gan_ind=0)
        for i in range(1, num_gans):
            g_states, g_actions, _ = imagination.test_trajectory(np.tile(curr_state, (num_trajs_per_gan, 1)),
                                                                 np.tile(end_goal, (num_trajs_per_gan, 1)), num_steps=max_steps,
                                                                 state_noise=state_noise, gan_ind=i)
            gen_states = np.concatenate((gen_states, g_states), axis=0)
            gen_actions = np.concatenate((gen_actions, g_actions), axis=0)

        success_mat = env.batch_goal_achieved(gen_states, end_goal[np.newaxis, np.newaxis, :], tol=tol)
        success_fraction = np.sum(success_mat, axis=-1) / max_steps
        success_inds = np.where(success_fraction>0)[0]
        if len(success_inds) == 0:
            return env.action_space.sample(), 0
        num_best = int(np.floor(frac_best*num_trajs))
        if len(success_inds) < num_best:
            num_best = len(success_inds)
        best_inds = (-success_fraction).argsort()[:num_best]
        success_frac_best = success_fraction[best_inds]

        num_reps_final_per_gan = num_reps_final // num_gans
        init_actions = gen_actions[best_inds, 0, :]
        start_states = np.repeat(gen_states[best_inds, 1, :], num_reps_final_per_gan, axis=0)
        end_goals = np.tile(end_goal, (start_states.shape[0], 1))

        scores = []
        for i in range(num_gans):
            gen_states_2, _, _ = imagination.test_trajectory(start_states, end_goals, num_steps=max_steps-1,
                                                             state_noise=state_noise, gan_ind=i)
            success_mat = env.batch_goal_achieved(gen_states_2, end_goals[:, np.newaxis, :], tol=tol)
            success_fraction_2 = np.sum(success_mat, axis=-1).reshape(num_best, num_reps_final_per_gan)
            success_fraction = np.concatenate((success_fraction_2, success_frac_best[:, np.newaxis]), axis=1)
            scores.append(np.sum(np.exp(alpha * success_fraction), axis=-1))

        scores = np.stack(scores)
        scores = np.mean(scores, axis=0)

        return init_actions[np.argmax(scores), :], 1




class TrajectoryFracPlannerRotInvariant2:
    def __init__(self, planning_args={}):
        self.planning_args = planning_args
        self.quaternion_invariants = np.concatenate(
            ((1.0 / np.sqrt(2)) * np.array([
                [1, 1, 0, 0],
                [1, -1, 0, 0],
                [1, 0, 1, 0],
                [1, 0, -1, 0],
                [1, 0, 0, 1],
                [1, 0, 0, -1]
            ]),
             np.array([
                 [0, 1, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1],
                 [0, -1, 0, 0],
                 [0, 0, -1, 0],
                 [0, 0, 0, -1],
                 [1, 0, 0, 0]
             ])), axis=0
        )  # invariants of cube.
        self.prev_angle = np.array([0.0, 0.0, 0.0])

    def generate_next_action(self, curr_state, end_goal, imagination, env, num_trajs=1000, max_steps=50,
                             frac_best=0.05, num_reps_final=250, alpha=1.0, state_noise=0.0, tol=0.05, return_average=False):
        state = env.save_state()
        quaternions = np.tile(state.qpos[-4:], (len(self.quaternion_invariants), 1))
        quaternion_rot = quat_mul(quaternions, self.quaternion_invariants)
        euler_angles = quat2euler(quaternion_rot)
        scores = np.linalg.norm(euler_angles[:, :2], axis=-1) + np.abs(
            euler_angles[:, -1] - np.tile(self.prev_angle[-1], len(euler_angles)))
        ind = np.argmin(scores)
        self.prev_angle = euler_angles[ind, :]
        state.qpos[-4:] = quaternion_rot[ind, :]
        state.qvel[-3:] = quat_rot_vec(quat_conjugate(self.quaternion_invariants[ind]), state.qvel[-3:])
        env.restore_state(state)
        curr_state = env._get_obs()["observation"]

        gen_states, gen_actions = imagination.test_traj_rand_gan(
            np.tile(curr_state, (num_trajs, 1)),
            np.tile(end_goal, (num_trajs, 1)), num_steps=max_steps
        )
        success_mat = env.batch_goal_achieved(gen_states, end_goal[np.newaxis, np.newaxis, :], tol=tol)
        success_fraction = np.sum(success_mat, axis=-1) / max_steps
        success_inds = np.where(success_fraction > 0)[0]
        if len(success_inds) == 0:
            return env.action_space.sample(), 0
        num_best = int(np.floor(frac_best * num_trajs))
        if len(success_inds) < num_best:
            num_best = len(success_inds)
        best_inds = (-success_fraction).argsort()[:num_best]
        success_frac_best = success_fraction[best_inds]
        init_actions = gen_actions[best_inds, 0, :]
        start_states = np.repeat(gen_states[best_inds, 1, :], num_reps_final, axis=0)
        end_goals = np.tile(end_goal, (start_states.shape[0], 1))
        #second iteration
        gen_states_2, _ = imagination.test_traj_rand_gan(
            start_states, end_goals, num_steps=max_steps-1
        )
        success_mat = env.batch_goal_achieved(gen_states_2, end_goals[:, np.newaxis, :], tol=tol)
        success_fraction_2 = np.sum(success_mat, axis=-1).reshape(num_best, num_reps_final) / max_steps
        success_fraction = np.concatenate((success_fraction_2, success_frac_best[:, np.newaxis]), axis=1)
        #scores = np.sum(np.exp(alpha * success_fraction), axis=-1)
        scores = np.exp(alpha*np.mean(success_fraction, axis=-1))
        if return_average:
            return np.sum(init_actions*scores[..., np.newaxis], axis=0) / np.sum(scores), 1
        else:
            return init_actions[np.argmax(scores), :], 1


class IterativePlanner(object):
    def __init__(self, planning_args={}):
        self.planning_args = planning_args
        self.quaternion_invariants = np.concatenate(
            ((1.0 / np.sqrt(2)) * np.array([
                [1, 1, 0, 0],
                [1, -1, 0, 0],
                [1, 0, 1, 0],
                [1, 0, -1, 0],
                [1, 0, 0, 1],
                [1, 0, 0, -1]
            ]),
             np.array([
                 [0, 1, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1],
                 [0, -1, 0, 0],
                 [0, 0, -1, 0],
                 [0, 0, 0, -1],
                 [1, 0, 0, 0]
             ])), axis=0
        )  # invariants of cube.
        self.prev_angle = np.array([0.0, 0.0, 0.0])

    def generate_next_action(self, curr_state, end_goal, imagination, env, num_acs=50, max_steps=50,
                             num_copies=100, num_reps=5, num_iterations=2, alpha=1.0, osm_frac=0.0,
                             tol=0.05, return_average=True, noise=0.1):
        state = env.save_state()
        quaternions = np.tile(state.qpos[-4:], (len(self.quaternion_invariants), 1))
        quaternion_rot = quat_mul(quaternions, self.quaternion_invariants)
        euler_angles = quat2euler(quaternion_rot)
        scores = np.linalg.norm(euler_angles[:, :2], axis=-1) + np.abs(
            euler_angles[:, -1] - np.tile(self.prev_angle[-1], len(euler_angles)))
        ind = np.argmin(scores)
        self.prev_angle = euler_angles[ind, :]
        state.qpos[-4:] = quaternion_rot[ind, :]
        state.qvel[-3:] = quat_rot_vec(quat_conjugate(self.quaternion_invariants[ind]), state.qvel[-3:])
        env.restore_state(state)
        curr_state = env._get_obs()["observation"]

        gen_states, gen_actions = imagination.test_traj_rand_gan(
            np.tile(curr_state, (num_acs, 1)),
            np.tile(end_goal, (num_acs, 1)), num_steps=1, frac_replaced_with_osm_pred=0.0
        )
        curr_states = gen_states[:, 0, :]
        curr_init_actions = gen_actions[:, 0, :]
        start_states = np.repeat(curr_states, num_copies, axis=0)
        end_goals = np.tile(end_goal, (start_states.shape[0], 1))
        num_osms = len(imagination.one_step_model.networks)
        best_action = None

        for it in range(num_iterations):
            inds = np.array_split(np.random.permutation(start_states.shape[0]), num_osms)
            rep_acs = np.repeat(curr_init_actions, num_copies, axis=0)

            success_fracs = []
            for j in range(num_reps):
                next_states = np.zeros_like(start_states)
                for i in range(num_osms):
                    next_states[inds[i],...] = imagination.one_step_model.predict(
                        start_states[inds[i],...], rep_acs[inds[i],...], osm_ind=i, normed_input=False
                    )
                gen_states, _ = imagination.test_traj_rand_gan(
                    next_states, end_goals, num_steps=max_steps-1, frac_replaced_with_osm_pred=osm_frac
                )
                success_mat = env.batch_goal_achieved(gen_states, end_goals[:, np.newaxis, :], tol=tol)
                success_fracs.append(np.sum(success_mat, axis=-1).reshape(num_acs, num_copies)/max_steps)
            success_fracs = np.concatenate(success_fracs, axis=1)
            pseudo_rewards = np.mean(success_fracs, axis=-1)
            succ_frac = pseudo_rewards.copy()

            if return_average:
                pseudo_rewards = np.exp(alpha * np.clip((pseudo_rewards - pseudo_rewards.mean()) / (pseudo_rewards.std() + 1e-4), -2.0, 2.0))
                best_action = np.sum(curr_init_actions*pseudo_rewards[:, np.newaxis], axis=0) / np.sum(pseudo_rewards)
            else:
                best_action = curr_init_actions[np.argmax(pseudo_rewards), :]

            curr_init_actions = np.tile(best_action, (num_acs,1))
            curr_init_actions[1:] = np.clip(curr_init_actions[1:] + noise*np.random.randn(*curr_init_actions[1:].shape), -1.0, 1.0)

        return best_action, 1


class SuperBasicPlanner:
    def __init__(self, planning_args={}):
        self.planning_args = planning_args
        self.quaternion_invariants = np.concatenate(
            ((1.0 / np.sqrt(2)) * np.array([
                [1, 1, 0, 0],
                [1, -1, 0, 0],
                [1, 0, 1, 0],
                [1, 0, -1, 0],
                [1, 0, 0, 1],
                [1, 0, 0, -1]
            ]),
             np.array([
                 [0, 1, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1],
                 [0, -1, 0, 0],
                 [0, 0, -1, 0],
                 [0, 0, 0, -1],
                 [1, 0, 0, 0]
             ])), axis=0
        )  # invariants of cube.
        self.prev_angle = np.array([0.0, 0.0, 0.0])

    def generate_next_action(self, curr_state, end_goal, imagination, env, num_acs=5000, noise=0.0):
        state = env.save_state()
        quaternions = np.tile(state.qpos[-4:], (len(self.quaternion_invariants), 1))
        quaternion_rot = quat_mul(quaternions, self.quaternion_invariants)
        euler_angles = quat2euler(quaternion_rot)
        scores = np.linalg.norm(euler_angles[:, :2], axis=-1) + np.abs(
            euler_angles[:, -1] - np.tile(self.prev_angle[-1], len(euler_angles)))
        ind = np.argmin(scores)
        self.prev_angle = euler_angles[ind, :]
        state.qpos[-4:] = quaternion_rot[ind, :]
        state.qvel[-3:] = quat_rot_vec(quat_conjugate(self.quaternion_invariants[ind]), state.qvel[-3:])
        env.restore_state(state)
        curr_state = env._get_obs()["observation"]

        init_states = np.tile(curr_state, (num_acs, 1))
        init_states += noise*np.random.randn(*init_states.shape)

        _, gen_actions = imagination.test_traj_rand_gan(
            init_states, np.tile(end_goal, (num_acs, 1)), num_steps=1
        )
        return np.mean(gen_actions[:,0,:], axis=0), 1