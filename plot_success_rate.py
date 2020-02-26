import joblib
import matplotlib.pyplot as plt

#expt_name = "FP_1000init_2kafter_OSMreg_newplanner_fullstate"
#eval_data = joblib.load("experiments/"+expt_name+"/eval.pkl")

"""
expt_names = [
    "FP_1000init_2kafter_OSMreg_newplanner_2fix",
    "FP_1000init_2kafter_OSMreg_newplanner_2fix_l2=30",
    "FP_1000init_2kafter_OSMreg_newplanner_2fix_l2=5",
    "FP_1000init_2kafter_OSMreg_newplanner_2fix_l2=1",
    "FP_1000init_2kafter_OSMreg_newplanner_smallbuffer",
    "FP_1000init_6kafter_OSMreg_newplanner_fullstate_l2=30.0",
    "FP_1000init_4kafter_OSMreg_newplanner_fullstate_l2=30.0_smallbuffer"
]

expt_legend_names = [
    "Reduced state, 1k init, 2k after, 500k init train. model_l2_reg=10.0",
    "Reduced state, 1k init, 2k after, 500k init train. model_l2_reg=30.0",
    "Reduced state, 1k init, 2k after, 500k init train. model_l2_reg=5.0",
    "Reduced state, 1k init, 2k after, 500k init train. model_l2_reg=1.0",
    "Reduced state, 1k init, 2k after, 500k init train. model_l2_reg=10.0. Small buffer (25k)",
    "Full state, 1k init, 6k after, 1 mil init train. model_l2_reg=30.0",
    "Full state, 1k init, 4k after, 1 mil init train. model_l2_reg=30.0. small buffer (25k)"
]
"""

"""
expt_names = [
    "FP_1000init_5kafter_OSMreg_newplanner_fullstate_l2=30.0_smallbuffer_50k_DS",
    "FP_250init_2kafter_OSMreg_newplanner_fullstate_l2=30.0_smallbuffer_25k",
    "FP_1000init_2kafter_OSMreg_newplanner_fullstate_l2=30.0_smallbuffer_50k",
    "FP_1000init_8retrain_1keach_OSMreg_newplanner_fullstate_l2=30.0_smallbuffer_100k",
    "FP_1000init_8retrain_1keach_OSMreg_newplanner_fullstate_l2=30.0_smallbuffer_100k_norefit"
]

expt_legend_names = [
    "Full state, 1k init, 5k after, buffer=50k, using DS",
    "Full state, 250 init, 2k after, buffer=25k",
    "Full state, 1k init, 2k after, buffer=50k",
    "Full state, 1k init, retrain every 1k (8 further times), buffer=100k, refit scalers",
    "Full state, 1k init, retrain every 1k (8 further times(, buffer=100k, "
]
"""

# expt_names = [
#     "FP_250init_2kafter_OSMreg_newplanner_fullstate_l2=30.0_smallbuffer_25k_randstart",
#     "FP_250init_2kafter_OSMreg_newplanner_fullstate_l2=30.0_smallbuffer_25k_randstart_correctedDS",
#     "FP_250init_2kafter_OSMreg_newplanner_fullstate_l2=30.0_smallbuffer_50k_randstart"
# ]
#
# expt_legend_names = [
#     "250 init, 2k extra, 25k buffer, randstart",
#     "250 init, 2k extra, 25k buffer, randstart, fuckedupDS",
#     "250 init, 2k extra, 50k buffer, randstart"
# ]

expt_names = [
    "FP_250init_2kafter_OSMreg_newplanner_fullstate_l2=30.0_smallbuffer_50k_explorationnoise=0.2_newsampling_randomfuturegoals_3GANS_3separateOSMs",
    "FP_250init_2kafter_OSMreg_newplanner_fullstate_l2=30.0_smallbuffer_50k_explorationnoise=0.2_newsampling_randomfuturegoals_3GANS_3OSMsmean",
]

expt_legend_names = [
    "3 GANS, 3 separate OSMs, 250 init, 2k extra, 50k buffer, l2=30.0, exploration noise=0.2, random future goals",
    "3 GANS, mean of 3 OSMs, 250 init, 2k extra, 50k buffer, l2=30.0, exploration noise=0.2, random future goals",
]

env_transitions = []
frac_success = []

for name in expt_names:
    data = joblib.load("experiments/"+name+"/eval.pkl")
    transitions = []
    success = []
    for datum in data:
        transitions.append(datum["num_env_transitions"])
        success.append(datum["frac_success"])
    env_transitions.append(transitions)
    frac_success.append(success)

for i in range(len(expt_names)):
    plt.plot(env_transitions[i], frac_success[i], '-o')

plt.legend(expt_legend_names)
plt.xlabel("Number of environment transitions")
plt.ylabel("Success fraction")

print("OK?")
