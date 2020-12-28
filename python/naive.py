import gym
import time

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

import argparse


def check_non_negative(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError("%s is an invalid non-negative int value" % value)
    return ivalue


def parse_args():
    parser = argparse.ArgumentParser(description='args parser')
    parser.add_argument('--seed', type=check_non_negative, help='seed value')

    args = parser.parse_args()
    return args


def generate_session(agent, max_iterations=10000, visualize=False):
    agent.eval()
    states, actions = [], []
    total_reward = 0
    s = env.reset()
    with torch.no_grad():
        for i in range(max_iterations):
            out = agent(torch.Tensor(s))
            a = np.random.choice(n_actions, p=F.softmax(out, dim=0).detach().numpy())
            action = a

            new_s, r, is_done, _ = env.step(action)
            if visualize:
                env.render()
            states.append(s)
            actions.append(action)
            total_reward += r
            s = new_s
            if is_done:
                break
    return states, actions, total_reward


def choose_elites(states, actions, rewards, p=30):
    states, actions = np.array(states), np.array(actions)
    perc = np.percentile(rewards, p)

    elite_actions = actions[rewards >= perc]
    elite_states = states[rewards >= perc]

    elite_states = np.concatenate(elite_states)
    elite_actions = np.concatenate(elite_actions)

    return elite_states, elite_actions


if __name__ == '__main__':
    args = parse_args()
    env = gym.make('CartPole-v0').env

    if args.seed:
        GLOBAL_SEED = args.seed
        np.random.seed(GLOBAL_SEED)
        env.seed(GLOBAL_SEED)
        torch.manual_seed(GLOBAL_SEED)

    s = env.reset()

    n_actions = env.action_space.n
    n_sessions = 100
    n_iterations = 100

    agent = nn.Sequential(nn.Linear(4, 15),
                          nn.ReLU(),
                          nn.Linear(15, 10),
                          nn.ReLU(),
                          nn.Linear(10, n_actions),
                          )

    optimizer = torch.optim.SGD(params=agent.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    for i in range(n_iterations):
        sessions = [generate_session(agent) for _ in range(n_sessions)]
        states, actions, total_rewards = zip(*sessions)
        elite_states, elite_actions = choose_elites(states, actions, total_rewards)
        print(f'Iteration: {i}, Mean Reward: {np.mean(total_rewards)}')
        # Train agent on elite
        agent.train()
        for s, a in zip(elite_states, elite_actions):
            a = a.item()
            target = torch.tensor([a])
            optimizer.zero_grad()
            out = agent(torch.Tensor(s).unsqueeze(0))
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

    generate_session(agent, visualize=True)
