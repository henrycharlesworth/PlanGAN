#!/bin/sh
python goal_conditioned_fromstart.py --noGP --addskipconnections --DS --b1=0.5 --b2=0.999 --num_discrim_updates=1 --l2reg_d --l2reg_g --l2regparam=0.0002 --experiment_name="from_start_betterparams_1"
