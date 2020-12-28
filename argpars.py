import argparse


def check_non_negative(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError("%s is an invalid non-negative int value" % value)
    return ivalue


def check_between_01(value):
    fvalue = float(value)
    if fvalue < 0 or fvalue > 1:
        raise argparse.ArgumentTypeError("%s is an invalid float. Should be value between [0, 1]" % value)
    return ivalue


def parse_args_cart_pole_dqn():
    parser = argparse.ArgumentParser(description='car_pole_dqn.py: CartPolev1 by DQN')
    parser.add_argument('--seed', type=check_non_negative, help='seed value')
    parser.add_argument('--test_runs', type=check_non_negative, default=25, help='number of test runs for check')
    parser.add_argument('--epsilon', type=check_between_01, default=0.4, help='epsilon value')
    parser.add_argument('--gamma', type=check_between_01, default=0.94, help='gamma value')
    parser.add_argument('--n_iterations', type=check_non_negative, default=10000, help='seed value')
    parser.add_argument('--max_iterations', type=check_non_negative, default=10000, help='seed value')
    parser.add_argument('--visualize',  action='store_true', help='visualize train and test runs')
    parser.add_argument('--logs', help='') # logs for tensorboard

    args = parser.parse_args()
    return args


def parse_args_mountain_car_dqn():
    parser = argparse.ArgumentParser(description='mountain_car_dqn.py: MountainCar-v0 by DQN')
    parser.add_argument('--seed', type=check_non_negative, help='seed value')
    parser.add_argument('--test_runs', type=check_non_negative, default=25, help='number of test runs for check')
    parser.add_argument('--gamma', type=check_between_01, default=0.95, help='gamma value')
    parser.add_argument('--epsilon', type=check_between_01, default=0.5, help='epsilon value')
    parser.add_argument('--n_iterations', type=check_non_negative, default=10000, help='seed value')
    parser.add_argument('--max_iterations', type=check_non_negative, default=10000, help='seed value')
    parser.add_argument('--visualize',  action='store_true', help='visualize train and test runs')
    parser.add_argument('--logs', help='') # logs for tensorboard

    args = parser.parse_args()
    return args
