#!/bin/sh

python learn_nearby_states_NEW_difftrajnums_wnoise.py --noGP --addskipconnections --DS --b1=0.5 --b2=0.999 --num_discrim_updates=1 --l2reg_d --l2reg_g --l2regparam=0.0002 --experiment_name="NEW_best_params_512_512_100trajsonly_l2_0_0002_noise_0_01" --h1=512 --h2=512 --num_trajs=100 --data_noise=0.01

python learn_nearby_states_NEW_difftrajnums_wnoise.py --noGP --addskipconnections --DS --b1=0.5 --b2=0.999 --num_discrim_updates=1 --l2reg_d --l2reg_g --l2regparam=0.0002 --experiment_name="NEW_best_params_512_512_100trajsonly_l2_0_0002_noise_0_05" --h1=512 --h2=512 --num_trajs=100 --data_noise=0.05

python learn_nearby_states_NEW_difftrajnums_wnoise.py --noGP --addskipconnections --DS --b1=0.5 --b2=0.999 --num_discrim_updates=1 --l2reg_d --l2reg_g --l2regparam=0.0002 --experiment_name="NEW_best_params_512_512_250trajsonly_l2_0_0002_noise_0_0" --h1=512 --h2=512 --num_trajs=250 --data_noise=0.0

python learn_nearby_states_NEW_difftrajnums_wnoise.py --noGP --addskipconnections --DS --b1=0.5 --b2=0.999 --num_discrim_updates=1 --l2reg_d --l2reg_g --l2regparam=0.0002 --experiment_name="NEW_best_params_512_512_250trajsonly_l2_0_0002_noise_0_05" --h1=512 --h2=512 --num_trajs=250 --data_noise=0.05
