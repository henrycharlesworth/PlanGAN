#!/bin/sh
python goal_conditioned_prevstate.py --noGP --addskipconnections --experiment_name="SNonly_skipconns_acnorm_5_tl_1000" --traj_length=1000 --action_norm=5.0
python goal_conditioned_prevstate.py --noGP --addskipconnections --experiment_name="SNonly_skipconns_acnorm_5_tl_500" --traj_length=500 --action_norm=5.0

