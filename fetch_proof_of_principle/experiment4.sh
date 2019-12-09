#!/bin/sh

python learn_nearby_states_NEW.py --noGP --addskipconnections --b1=0.5 --b2=0.999 --num_discrim_updates=1 --experiment_name="new_bestSNparams_SNpaper_2"
python learn_nearby_states_NEW.py --noGP --addskipconnections --b1=0.5 --b2=0.999 --num_discrim_updates=1 --DS --experiment_name="new_bestSNparams_SNpaper_DS"

python learn_nearby_states_NEW.py --noGP --addskipconnections --num_discrim_updates=1 --experiment_name="new_SNonly_skipconns_numdisupd=1"

python learn_nearby_states_NEW.py --noGP --addskipconnections --b1=0.5 --b2=0.999 --experiment_name="new_SNonly_skipconns_diffbetas"


