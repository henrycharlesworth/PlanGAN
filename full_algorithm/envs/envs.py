import abc

class Environment(object):

    @abc.abstractmethod
    def batch_goal_achieved(self, current, target, multiple_curr=False, multiple_target=False, final_goal=False):
        pass

    @abc.abstractmethod
    def get_goal_from_state(self, state, final_goal=False):
        pass