import gym
import numpy as np
import torch
from torch import nn


def generate_sessions(agent, n_session=100, max_iterations=10000, visualize=False):
    states, actions, total_reward = [], [], 0
    s = env.reset()
    for _ in range(max_iterations):
        out = agent(torch.Tensor(s)) 
        a = out.argmax().item() # come from agent
        action = a
        # res = np.random.choice([0, 1], p=[0.8, 0.2])
        # if res == 0:
        #     action = a
        # else:
        #     action = np.random.choice([0, 1])

        new_s, reward, is_done, _ = env.step(action)
        if visualize:
            env.render()
        states.append(s)
        actions.append(action)
        total_reward += reward
        s = new_s
        if is_done:
            break
    return states, actions, total_reward

def choose_elites(states, actions, rewards, p=60):
    perc = np.percentile(rewards, p)
    elite_actions = actions[rewards > perc]
    elite_states = states[rewards > perc]
    return elite_states, elite_actions


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    s = env.reset()
    n_actions = env.action_space.n
    n_sessions = 10
    n_iterations = 200
    print(n_actions)
    print(s)
    agent = nn.Sequential(
        nn.Linear(4, 15), 
        nn.ReLU(), 
        nn.Linear(15, 10),
        nn.ReLU(),
        nn.Linear(10, n_actions)
        )
    print(agent(torch.Tensor(s)))
    optimizer = torch.optim.SGD(params=agent.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    agent.train()
    for i in range(n_iterations):
        sessions = [generate_sessions(agent) for _ in range(n_sessions)]
        states, actions, total_reward = zip(*sessions)
        elite_states, elite_actions = choose_elites(np.array(states), np.array(actions), np.array(total_reward))
        print(f"Iteration {i}, mean reward: {np.mean(total_reward)}")
        for s, a in zip(elite_states, elite_actions):
            optimizer.zero_grad()
            out = agent(torch.Tensor(s))
            loss = criterion(out, torch.LongTensor(a))
            loss.backward()
            optimizer.step()
    generate_sessions(agent, visualize=True)
