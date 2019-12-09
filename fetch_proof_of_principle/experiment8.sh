#!/bin/sh
python learn_nearby_states_NEW.py --noGP --addskipconnections --DS --b1=0.5 --b2=0.999 --num_discrim_updates=1 --l2reg_d --l2reg_g --l2regparam=0.0002 --experiment_name="NEW_best_params_512_512" --h1=512 --h2=512

python learn_nearby_states_NEW.py --noGP --addskipconnections --DS --b1=0.5 --b2=0.999 --num_discrim_updates=1 --l2reg_d --l2reg_g --l2regparam=0.0002 --experiment_name="NEW_best_params_256_256" --h1=256 --h2=256


