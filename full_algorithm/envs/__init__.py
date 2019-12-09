from full_algorithm.envs.fetch_reach import FetchReach
import joblib
import os

ENV_LIST = ["fetch_reach", "fetch_reach_rand"]

def return_environment(env_name):
    print(os.getcwd())
    if env_name == "fetch_reach_rand":
        possible_states = joblib.load("envs/fetch_reach_start_states.pkl")
        return FetchReach(random_start=True, possible_start_states=possible_states)
    elif env_name == "fetch_reach":
        return FetchReach()
    else:
        raise ValueError("Invalid environment requested")