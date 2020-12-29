# rl_intro_labs

# Getting started
Storage for university labs use (RL).

## Dependecies
* Python version >= 3.8
* numpy >= 19.1
* gym 0.18.2
* pytorch >= 3.8.5
* tensorboard >= 2.4.0

## Tasks

### Tasks

* See first task on `task1.txt`
* All srcs from course on `python` dir
* all the the rest srcs for the second and third tasks:
  * DQN for `CartPole-v1` (Task #2)
  * DQN for `MountainCar-v0` (Task #3)
* Additionally added A2C for `MountainCarContinuous-v0`:
  * `experim_mountain_car_continous_a2c_1.py` that uses `a2c.py`
  * based on Yuxi Liu PyTorch 1.x Reinforcement Learning Cookbook samples

## Runnnig
* Install all dependecies on your env
* Run on the following commands
``` bash
python cart_pole_dqn_1.py --visualize
python mountain_car_dqn_1.py --visualize
```
   or just choose your own number for test runs
```bash
python mountain_car_dqn_1.py --visualize --test_runs=35
python cart_pole_dqn_1.py --visualize --test_runs=35
```

* For `MountainCarContinuous-v0` by A2C (additional case) run:
```bash
python experim_mountain_car_continous_a2c_1.py
```
or
```bash
python experim_mountain_car_continous_a2c_1.py --visualize --logs
```

## Note
* if the test runs fail by the end of train and agent cannot successfully complete test runs, the number that were specified, please, restart (and increase the number of iterations if necessary)
* recommended at least 6000 iters for `MountainCar-v0` env and 5000 for `CartPole-v1` env
* See ```python mountain_car_dqn_1.py --help``` or ```python cart_pole_dqn_1.py --help``` for help.
* `experim_mountain_car_continous_a2c_1.py` not stable.
