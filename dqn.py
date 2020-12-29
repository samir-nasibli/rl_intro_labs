import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class DQNAgent(nn.Module):
    def __init__(self, epsilon, gamma, n_actions, n_states):
        super(DQNAgent, self).__init__()
        self.epsilon = epsilon
        self.gamma = gamma
        self.n_actions = n_actions
        self.n_states = n_states

        self.fc1 = nn.Linear(n_states, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_best_action(self, a):
        return np.argmax(self(a).detach().cpu().numpy(), axis=1)

    def get_action(self, state):
        r = np.random.choice([0, 1], p=[self.epsilon, 1 - self.epsilon])
        if r == 0:
            a = np.array(np.random.choice(self.n_actions)).reshape(state.shape[0])
        else:
            a = self.get_best_action(state)
        return a


class DQNLoss(nn.Module):
    def __init__(self, gamma):
        super(DQNLoss, self).__init__()
        self.loss = nn.MSELoss()
        self.gamma = gamma

    def forward(self, agent, prev_agent_state, st, a, r, sts, D):
        pred_Q = torch.gather(agent(st), 1, a.reshape(-1, 1)).reshape(-1)
        with torch.no_grad():
            max_Q = prev_agent_state(sts).max(dim=1)[0]

        return self.loss(pred_Q, r + self.gamma * max_Q * D)
