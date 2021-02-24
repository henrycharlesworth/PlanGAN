# PlanGAN

Code for our NeurIPS 2020 paper ["PlanGAN"](https://proceedings.neurips.cc/paper/2020/file/6101903146e4bbf4999c449d78441606-Paper.pdf)

Requirements:
* Mujoco-py
* PyTorch
* NumPy
* SkLearn
* Joblib

To train an agent on FetchPickAndPlace run:

```python train.py --env="fetch_pick_and_place" --expt_name="FPAP"```

Pre-trained agents for FetchPush and FetchPickAndPlace are included in this repo. They can be evaluated with:

```python evaluate.py --expt_name="FetchPickAndPlace" --num_trajectories=50```

We also include a Jupyter notebook that allows you to visualise the imagined trajectories that the GANs generate (visualise_trajectories.ipynb).
