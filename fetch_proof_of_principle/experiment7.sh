#!/bin/sh
python learn_nearby_states_NEW.py --noGP --addskipconnections --DS --b1=0.5 --b2=0.999 --num_discrim_updates=1 --l2reg_d --l2reg_g --l2regparam=0.0002 --experiment_name="new_SNonly_skipconns_bestparams_DS_l2reg_1"
python learn_nearby_states_NEW.py --noGP --addskipconnections --DS --b1=0.5 --b2=0.999 --num_discrim_updates=1 --l2reg_d --l2reg_g --l2regparam=0.0002 --experiment_name="new_SNonly_skipconns_bestparams_DS_l2reg_2"
python learn_nearby_states_NEW.py --noGP --addskipconnections --DS --b1=0.5 --b2=0.999 --num_discrim_updates=1 --l2reg_d --l2reg_g --l2regparam=0.0002 --experiment_name="new_SNonly_skipconns_bestparams_DS_l2reg_3"
python learn_nearby_states_NEW.py --noGP --addskipconnections --DS --b1=0.5 --b2=0.999 --num_discrim_updates=1 --l2reg_d --l2reg_g --l2regparam=0.0002 --experiment_name="new_SNonly_skipconns_bestparams_DS_l2reg_4"

