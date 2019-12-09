import numpy as np

cg_goal_threshold = 0.5

def batch_goal_achieved(current, target, multiple_target=False, final_goal=False):
    if final_goal:
        pass
        #current = self.get_goal_from_state(current, final_goal=True)
        #target = self.get_goal_from_state(target, final_goal=True)
    if multiple_target:
        n_curr = current.shape[0]
        n_tar = target.shape[0]
        target = np.tile(target, (n_curr, 1))
        current = np.repeat(current, n_tar, axis=0)
        dists = np.linalg.norm(current - target, axis=-1)
        if final_goal:
            pass
            #success = dists < self.final_goal_threshold
        else:
            success = dists < cg_goal_threshold
        success = success.reshape(n_curr, n_tar)
        final_success = np.sum(success, axis=1)
        which_target = np.zeros((n_curr,)).astype(int)
        inds = np.where(final_success)[0]
        for ind in inds:
            which_target[ind] = np.random.choice(np.where(success[ind, :])[0])
        return final_success, which_target
    else:
        if len(current.shape) == 1 or (len(current.shape) == 2 and current.shape[0] == 1):
            if len(target.shape) == 1 or (len(target.shape) == 2 and target.shape[0] == 1):
                dists = np.linalg.norm(current - target)
            else:
                current = np.tile(current, (target.shape[0], 1))
                dists = np.linalg.norm(current - target, axis=-1)
        else:
            if len(target.shape) == 1 or (len(target.shape) == 2 and target.shape[0] == 1):
                target = np.tile(target, (current.shape[0], 1))
                dists = np.linalg.norm(current - target, axis=-1)
            else:
                dists = np.linalg.norm(current - target, axis=-1)
        if final_goal:
            pass
            #return dists < self.final_goal_threshold
        else:
            return dists < cg_goal_threshold

test_1 = np.array([0.1, 0.2, 0.6])
test_2 = np.array([0.2, 0.2, 0.6])
test_3 = np.array([[0.2, 0.2, 0.6]])

a1= batch_goal_achieved(test_1, test_2)
a2=batch_goal_achieved(test_1, test_3)

test_4 = np.array([[0.2, 0.2, 0.6],
                   [1.0, 0.3, 0.4],
                   [5.0, 0.2, 0.6]])

test_5 = np.array([[0.3, 0.2, 0.5],
                   [2.0, 0.3, 0.9],
                   [5.1, 0.1, 0.5]])

a3 = batch_goal_achieved(test_1, test_4)

a4 = batch_goal_achieved(test_4, test_1)

a5 = batch_goal_achieved(test_5, test_4)

test_6 = np.array([[0.3, 0.4, 0.6],
                   [0.9, 0.3, 0.4],
                   [20.0, 21.0, 22.0],
                   [5.1, 0.2, 0.6]])

a6, which_target_1 = batch_goal_achieved(test_4, test_6, multiple_target=True)

a7, which_target_2 = batch_goal_achieved(test_6, test_4, multiple_target=True)

print("OK")

#all appears to be working correctly.