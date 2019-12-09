#!/bin/sh

python learn_nearby_states_NEW_difftrajnums.py --noGP --addskipconnections --DS --b1=0.5 --b2=0.999 --num_discrim_updates=1 --l2reg_d --l2reg_g --l2regparam=0.0002 --experiment_name="NEW_best_params_512_512_100trajsonly" --h1=512 --h2=512 --num_trajs=100

python learn_nearby_states_NEW_difftrajnums.py --noGP --addskipconnections --DS --b1=0.5 --b2=0.999 --num_discrim_updates=1 --l2reg_d --l2reg_g --l2regparam=0.0002 --experiment_name="NEW_best_params_512_512_1000trajsonly" --h1=512 --h2=512 --num_trajs=1000

python learn_nearby_states_NEW_difftrajnums.py --noGP --addskipconnections --DS --b1=0.5 --b2=0.999 --num_discrim_updates=1 --l2reg_d --l2reg_g --l2regparam=0.002 --experiment_name="NEW_best_params_512_512_100trajsonly_l2_0_002" --h1=512 --h2=512 --num_trajs=100
