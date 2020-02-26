from envs.fetch_reach import FetchReach
from envs.fetch_push import FetchPush, FetchPushRandomStart

ENV_LIST = ["fetch_reach_reduced", "fetch_reach", "fetch_push", "fetch_push_reduced", "fetch_push_ng",
            "fetch_push_ng_rand"]

def return_environment(env_name):
    if env_name == "fetch_reach":
        return FetchReach()
    elif env_name == "fetch_reach_reduced":
        return FetchReach(reduced=True)
    elif env_name == "fetch_push":
        return FetchPush()
    elif env_name == "fetch_push_reduced":
        return FetchPush(reduced=True)
    elif env_name == "fetch_push_ng":
        return FetchPush(remove_gripper=True)
    elif env_name == "fetch_push_ng_rand":
        return FetchPushRandomStart(remove_gripper=True)
    else:
        raise ValueError("Invalid environment requested!")