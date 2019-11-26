#!/bin/sh

#python learn_nearby_states_NEW.py --noGP --addskipconnections --experiment_name="new_SNonly_skipconns_3tanhgen" --generator_tanh --generator_tanh_value=3.0
python learn_nearby_states_NEW.py --noGP --addskipconnections --experiment_name="new_SNonly_skipconns_DS_3tanhgen" --generator_tanh --generator_tanh_value=3.0 --DS
python learn_nearby_states_NEW.py --noGP --addskipconnections --experiment_name="new_SNonly_skipconns_DS_altnorm" --alternative_ds_normalisation --DS
python learn_nearby_states_NEW.py --noGP --addskipconnections --experiment_name="new_SNonly_skipconns_DS" --DS
