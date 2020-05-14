from envs.fetch_reach import FetchReach
from envs.fetch_pick_and_place import FetchPickAndPlace
from envs.fetch_push import FetchPush
from envs.four_rooms.return_env import return_standard_four_rooms
from envs.reacher.reacher_three import ReacherThreeEnv

ENV_LIST = ["fetch_reach_reduced", "fetch_reach", "fetch_push", "fetch_pick_and_place", "fetch_slide_ng", "four_rooms", "reacher_three"]

def return_environment(env_name):
    if env_name == "fetch_reach":
        return FetchReach()
    elif env_name == "fetch_reach_reduced":
        return FetchReach(reduced=True)
    elif env_name == "fetch_push":
        return FetchPush(remove_gripper=True)
    elif env_name == "fetch_pick_and_place":
        return FetchPickAndPlace()
    elif env_name == "four_rooms":
        return return_standard_four_rooms()
    elif env_name == "reacher_three":
        return ReacherThreeEnv()
    else:
        raise ValueError("Invalid environment requested!")
