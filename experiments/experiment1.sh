#!/bin/sh

cd ..

CUDA_VISIBLE_DEVICES=2 python main.py --expt_name="FP_250init_2kafter_OSMreg_newplanner_fullstate_l2=30.0_smallbuffer_50k_explorationnoise=0.2_newsampling_randomfuturegoals_3GANS_3separateOSMs" --env="fetch_push_ng" --init_rand_trajs=250 --init_gan_train_its=250000 --init_osm_train_its=250000 --filter_rand_trajs --reg_gan_with_osm --gan_model_l2=30.0 --l2_G=0.001 --l2_D=0.001 --gan_train_end=250000 --osm_train_end=100000 --extra_trajs=2000 --planner_type="trajfrac" --train_per_extra_gan=500 --buffer_capacity=50000 --exploration_noise=0.2 --filter_train_batch --random_future_goals --num_gans=3 --num_osms=3
