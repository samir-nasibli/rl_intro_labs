import gym
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import time


def generate_session(agent, max_iterations=10000, visualize=False):
    agent.eval()
    states, actions, rewards = [], [], []
    s = env.reset()
    with torch.no_grad():
        for i in range(max_iterations):
            out = agent(torch.Tensor(s))
            action = np.random.choice(n_actions, p=F.softmax(out, dim=0).detach().numpy())
            new_s, r, is_done, _ = env.step(action)
            if visualize:
                env.render()
                time.sleep(0.02)
            states.append(s)
            actions.append(action)
            rewards.append(r)

            s = new_s
            if is_done:
                break
    return states, actions, rewards


def get_q_estimate(rewards, gamma=0.99):
    q_estimates = []
    moving_sum = 0
    for r in reversed(rewards):
        current = r + gamma * moving_sum
        moving_sum = current 
        q_estimates.append(current)
    return reversed(q_estimates)


class ReinforceAgent(nn.Module):
    def __init__(self, n_actions, state_dim):
        super(ReinforceAgent, self).__init__()
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.fc1 = nn.Linear(self.state_dim[0], 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(agent):
    optimizer = optim.Adam(agent.parameters(), lr=0.001)
    n_sessions = 100
    n_iterations = 1000

    for i in range(n_iterations):
        for j in range(n_sessions):
            states, actions, rewards = generate_session(agent)
            agent.train()
            states = torch.Tensor(states)
            q_estimates = torch.Tensor(list(get_q_estimate(rewards)))
            logits = agent(states)
            log_probabilities = F.log_softmax(logits, dim=1)

            temp = torch.sum(log_probabilities * F.one_hot(torch.LongTensor(actions)), dim=1)
     
            objective = torch.mean(temp * q_estimates)
            loss = - objective
            loss.backward()
            optimizer.step()
        print(f"Iteration: {i}, Rewards: {np.sum(rewards)}")
        if np.sum(rewards) > 300:
            break


if __name__ == '__main__':
    env = gym.make('CartPole-v0').env
    s = env.reset()

    n_actions = env.action_space.n
    state_dim = env.observation_space.shape

    agent = ReinforceAgent(n_actions, state_dim)

    train(agent)

    generate_session(agent, visualize=True)
