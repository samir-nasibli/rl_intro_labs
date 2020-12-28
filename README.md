# rl_intro_labs

# Getting started
Storage for university labs use (RL).

## Dependecies
* Python version >= 3.8
* numpy >= 19.1
* gym 0.18.2
* pytorch >= 3.8.5

## Tasks

### Tasks

* See first task on `task1.txt`
* All srcs from course on `python` dir
* all the the rest srcs for the second and third tasks:
  * DQN for `CartPole-v1` (Task #2)
  * DQN for `MountainCar-v0` (Task #3)

## Runnnig
* Install all dependecies on your env

``` bash
python cart_pole_dqn_1.py --visualize
python mountain_car_dqn_1.py --visualize
```

## Note
* if the test runs fail try to increase the number of iterations for training. 
See, ```python mountain_car_dqn_1.py --help``` or ```python cart_pole_dqn_1.py --help```

## TODO
* add logs for tensorboard
* add tensorboard to dependencies
