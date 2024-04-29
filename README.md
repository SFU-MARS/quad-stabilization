# Quadrotor Stabilization
This repository contains the implementation of the robust quadrotor stabilization research work.

## Installation
The simulation environment is based on the [OptimizedDP library](https://github.com/SFU-MARS/optimized_dp) and the [phoenix-drone-simulation](https://github.com/SvenGronauer/phoenix-drone-simulation.git).

0. Install the [pytorch](https://pytorch.org/).

1. First, run the following command to create a virtual environment named quadrotor and install the required packages (this process would take some time):
``conda env create -f environment.yml``

2. Then install the odp package (from the Optimized_DP library ) by:
```
cd adversarial_generation
pip install -e.
cd ..
```

3. One method to install the phoenix-drone-simulation package is to comment out the `mpy4pi` in the `install_requires` in the file `phoenix_drone_simulation/setup.py`, and use the following commands:
```
$ git clone https://github.com/SvenGronauer/phoenix-drone-simulation
$ cd phoenix-drone-simulation/
$ pip install -e .
```

The other method is to use `git submodule add` command. Unfortunately, I have not tried this before so I can not give any advice.

4. Finally install other dependencies:
```
pip install joblib
pip install tensorboard
conda install mpy4pi [reference](https://stackoverflow.com/questions/74427664/error-could-not-build-wheels-for-mpi4py-which-is-required-to-install-pyproject)
conda install pandas
```

Sorry for the complex installations, we will sort them up later.

# Working Logs
2023.8.23
The trying of the env_id `DroneHoverBulletEnvWithAdversary-v0` failed (both PPO and SAC, both PPO from sb3 and PPO here). 
Now try the origional command `python -m phoenix_drone_simulation.train --alg ppo --env DroneHoverBulletEnv-v0` in the tmux environment `phonex_base`. Try to make it clear how the algorithm works and where to store the training results, how to display using the stored checkpoints. Then try our environment with its original local PPO.

I train our env with no distb in tmux `phoenix_nodistb_oppo`, command: `python adversary_phoenix.py`, pay attention to the save log directory.

Results:
| environment | algorithm | train logdir | test logdir | performance |
| ------------|-----------|------------|-----------| ----------- |
|DroneHoverBulletEnvWithAdversary-v0 with no dstb | original PPO |  runs/original_ppo/no_distb/ | tests/phoenix/random_d/.../2023-08-24__11-48-40| Good |
| DroneHoverBulletEnv-v0 | original PPO | runs/base | python -m phoenix_drone_simulation.play --ckpt runs/base/ppo/2023-08-23__15-58-26/seed_36626 | OK |
| DroneHoverBulletEnvWithAdversary-v0 with no dstb | sb3 PPO | sb3_tensorboard/DroneHoverBulletEnvWithAdversary-v0/08_24_15_04 | None | less OK |
| DroneHoverBulletEnvWithAdversary-v0 with random dstb | sb3 PPO | Unknow | None | Unknow| 
| DroneHoverBulletEnvwithAdersary-v0 with random dstb | original PPO (python -m phoenix_drone_simulation.train --alg ppo --env DroneHoverBulletEnvwithAdersary-v0) | unknow | unknow
| DroneHoverBulletEnvWithAdversary-v0 with HJ dstb | sb3 PPO | Unknow | None | Unknow| 
| DroneHoverBulletEnvwithAdersary-v0 with HJ dstb | original PPO (python adversaryhover_phoenix.py) | unknow | unknow

2023.8.24
Add our custermized environment `DroneHoverBulletEnvWithAdversary-v0` to its original register codes (`phoenix-drone-simulation/phoenix_drone_simulation/__init__.py`) so we do not need to register this environment everytime we use it. The details of the code are in the `phoenix-drone-simulation/phoenix_drone_simulation/envs/hover.py`.

2023.8.27
Try command `python -m phoenix_drone_simulation.train --alg ppo --env DroneHoverBulletEnvWithAdversary-v0 --log-dir /localhome/hha160/projects/quadrotor-stabilization/runs/original_ppo `. 
This works but it takes so long. So just stop.

2023.8.28
Test command: python -m phoenix_drone_simulation.play --ckpt PATH_TO_CKPT; # Interesting thing: this test env seems to be different of our codes, like `test_hhy.py` etc. Why?
Train with our adversaryhover_phoenix.py: `python adversaryhover_phoenix.py`; 
Compare it with `python -m phoenix_drone_simulation.train --alg ppo --env DroneHoverBulletEnvWithAdversary-v0 --log-dir /localhome/hha160/projects/quadrotor-stabilization/runs/original_ppo/our_distb` (not start yet today). # I think the later one will have a better performance because of the usage of the parallel tool `mpi`.


## Transfer sb3
Xubo's advice:
1. Pay attention to the range of the action space while using sb3 algorithms!
If the range is not the same, try to multiply a constant coefficient!
2. Could also check https://github.com/gsilano/CrazyS

## Framework Outline
