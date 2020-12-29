import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

import sklearn.preprocessing

from torch.utils.tensorboard import SummaryWriter

import numpy as np

from a2c import (ActorCriticModel, PolicyNetwork)

from argpars import parse_args_mountain_car_a2c


def actor_critic(env, estimator, n_episode, gamma=1.0, logs=False, visualize=False):
    """
    continuous Actor Critic algorithm
    @param env: Gym environment
    @param estimator: policy network
    @param n_episode: number of episodes
    @param gamma: the discount factor

    """
    if logs:
        tb = SummaryWriter()
    total_reward_episode = [0] * n_episode
    for episode in range(n_episode):
        log_probs = []
        rewards = []
        state_values = []
        state = env.reset()
        while True:
            state = scale_state(state)
            action, log_prob, state_value =  estimator.get_action(state)
            action = action.clip(env.action_space.low[0],
                                 env.action_space.high[0])
            next_state, reward, is_done, _ = env.step(action)
            total_reward_episode[episode] += reward
            log_probs.append(log_prob)
            state_values.append(state_value)
            rewards.append(reward)
            if is_done:
                returns = []
                Gt = 0
                pw = 0
                for reward in rewards[::-1]:
                    Gt += gamma ** pw * reward
                    pw += 1
                    returns.append(Gt)
                returns = returns[::-1]
                returns = torch.tensor(returns)
                returns = (returns - returns.mean()) /  (returns.std() + 1e-9)
                estimator.update( returns, log_probs, state_values)
                if logs:
                    tb.add_scalar("MountainCarContinuous_a2c/total_reward", total_reward_episode[episode], episode)
                print('Episode: {}, total reward: {}'.format( episode, total_reward_episode[episode]))
                break
            state = next_state
            if visualize:
                env.render()


def scale_state(state):
    scaled = scaler.transform([state])
    return scaled[0]


if __name__ == '__main__':
    args = parse_args_mountain_car_a2c()
    env = gym.make('MountainCarContinuous-v0')
    env.reset()

    gamma = args.gamma # 0.9
    n_episode = args.n_episode
    logs = args.logs
    visualize = args.visualize

    state_space_samples = np.array([env.observation_space.sample() for x in range(10000)])
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(state_space_samples)

    n_state = env.observation_space.shape[0]
    n_action = 1
    n_hidden = 128
    lr = 0.0003

    policy_net = PolicyNetwork(n_state, n_action, n_hidden, lr)
    actor_critic(env, policy_net, n_episode, gamma, logs, visualize)
    env.close()
