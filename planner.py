import numpy as np

from gym.envs.robotics.rotations import quat_mul, euler2quat, quat2euler, mat2euler, quat_rot_vec, quat_conjugate

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
        )  # invariants of cube. For FP and FPAP (not necessary, actually).
        self.reset()

    def reset(self):
        self.prev_angle = np.array([0.0, 0.0, 0.0])

    def generate_next_action(self, curr_state, end_goal, imagination, env, num_acs=50, max_steps=50,
                             num_copies=100, num_reps=5, num_iterations=2, alpha=1.0, osm_frac=0.0,
                             tol=0.05, return_average=True, noise=0.1):
        if env.name.startswith("fetch_push") or env.name.startswith("fetch_pick"):
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
                #pseudo_rewards = np.exp(alpha * np.clip((pseudo_rewards - pseudo_rewards.mean()) / (pseudo_rewards.std() + 1e-4), -2.0, 2.0))
                pseudo_rewards = (pseudo_rewards - pseudo_rewards.mean()) / (pseudo_rewards.std() + 1e-4)
                pseudo_rewards = np.exp(alpha*(pseudo_rewards - pseudo_rewards.max()))
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

    def generate_next_action(self, curr_state, end_goal, imagination, env, num_acs=5000, noise=0.0, average=True):

        if average == False:
            num_acs = 3

        if env.name.startswith("fetch_push") or env.name.startswith("fetch_pick"):
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
        if average:
            return np.mean(gen_actions[:,0,:], axis=0), 1
        else:
            return gen_actions[0, 0, :], 1