#!/bin/sh
python learn_nearby_states_NEW.py --noGP --addskipconnections --experiment_name="new_SNonly_skipconns_2"
python learn_nearby_states_NEW.py --noGP --addskipconnections --experiment_name="new_SNonly_skipconns_3"
python learn_nearby_states_NEW.py --noGP --addskipconnections --experiment_name="new_SNonly_skipconns_4"

python learn_nearby_states_NEW.py --noGP --experiment_name="new_SNonly_standarch_2"
python learn_nearby_states_NEW.py --noGP --experiment_name="new_SNonly_standarch_3"
python learn_nearby_states_NEW.py --noGP --experiment_name="new_SNonly_standarch_4"

python learn_nearby_states_NEW.py --experiment_name="new_SN_standarch_2"
