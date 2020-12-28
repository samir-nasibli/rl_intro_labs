import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import random

from dqn import (DQNAgent, DQNLoss)
from rl_tools import Experience

from argpars import parse_args_mountain_car_dqn


def generate_session(agent, max_iterations=10000, gamma=0.9, visualize=False):
    agent.eval()
    states, actions, new_states, rewards, D = [], [], [], [], []
    total_reward = 0
    s = env.reset()
    gamma_reward = 0
    max_height = 0
    with torch.no_grad():
        for i in range(max_iterations):
            action = (agent.get_action(torch.from_numpy(np.array(s)).reshape(1,-1).float())).item()
            new_s, r, is_done, _ = env.step(action)

            if visualize:
                env.render()

            D.append(1 - int(is_done))
            states.append(s)
            new_states.append(new_s)
            actions.append(action)
            height = (new_s[0] + 0.5)**2 
            if height > max_height :
                max_height = height
                r += height * 15
            total_reward += r
            if is_done and i < 199:
                r = 250
            elif is_done:
                r += -25
            rewards.append(r)
            gamma_reward += r * gamma ** i
            s = new_s
            if is_done:
                break
    return states, actions, new_states, rewards,D, np.array([gamma_reward])


def train(agent, n_iterations, max_iterations, visualize, test_runs):
    experience = Experience()
    prev_agent_state = copy.deepcopy(agent)
    optimizer = torch.optim.Adam(params=agent.parameters(), lr=0.002)
    criterion = DQNLoss(gamma)

    max_reward = 0
    for i in range(n_iterations):
        optimizer.zero_grad()
        s = env.reset()
        total_reward = 0.0

        states, actions, new_states, rewards ,D, total_reward = generate_session(agent, max_iterations, gamma=gamma)

        experience.add(states, actions, new_states, rewards,D)
        states, actions, new_states, rewards, D = experience.replay()

        states = torch.from_numpy(np.array(states)).float()
        new_states = torch.from_numpy(np.array(new_states)).float()
        actions = torch.from_numpy(np.array(actions)).long()
        rewards = torch.from_numpy(np.array(rewards)).float()
        D = torch.from_numpy(np.array(D)).float()

        loss = criterion(agent, prev_agent_state, states, actions, rewards, new_states, D)
        loss.backward()
        optimizer.step()
        total_reward = total_reward.max()
        if max_reward < total_reward:
            max_reward = total_reward
            print(max_reward)
        if i % 100 == 0:
            total_reward = 0
            prev_eps = agent.epsilon
            agent.epsilon = 0
            testing_status = True
            test_run = 0
            while testing_status:
                s = env.reset()
                for iters in range(max_iterations):
                    a = (agent.get_action(torch.from_numpy(np.array(s)).reshape(1, -1).float())).item()
                    s_new, r, is_done, _ = env.step(a)
                    total_reward += r
                    if is_done and iters >= 199:
                        testing_status = False
                    if is_done:
                        break
                    s = s_new
                    if visualize:
                        env.render()
                print(f'test run {test_run} is {testing_status}')
                test_run += 1
                if not testing_status:
                    break
                if test_run >= test_runs:
                    testing_status = True
                    break
            if testing_status:
                    break 
            agent.epsilon = prev_eps
            agent.epsilon = (agent.epsilon - 0.0025) if agent.epsilon - 0.0025 > 0 else 0
            prev_agent_state.load_state_dict(agent.state_dict())
            print(f'Iteration: {i}, Total reward: {total_reward}, Max reward: {max_reward},  Epsilon: {agent.epsilon}')


if __name__ == '__main__':
    args = parse_args_mountain_car_dqn()
    env = gym.make('MountainCar-v0')

    visualize = args.visualize
    test_runs = args.test_runs

    gamma = args.gamma # 0.5
    epsilon = args.epsilon # 0.4
    n_iterations = args.n_iterations # 10000
    max_iterations = args.max_iterations # 10000

    #if args.seed:
    #    GLOBAL_SEED = args.seed
    #    np.random.seed(GLOBAL_SEED)
    #    env.seed(GLOBAL_SEED)
    #    torch.manual_seed(GLOBAL_SEED)

    env.reset()

    if visualize:
        env.render()
    
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n

    print(f'States number: {n_states}, Actions number: {n_actions}')

    s_new, reward, is_done, _ = env.step(0)

    agent = DQNAgent(epsilon, gamma, n_actions, n_states)

    train(agent, n_iterations, max_iterations, visualize, test_runs)
