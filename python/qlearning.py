import gym
import numpy as np


class QLearningAgent:
    def __init__(self, alpha, epsilon, gamma, n_actions, n_states):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.n_actions = n_actions
        self.n_states = n_states
        self.qvalues = np.zeros((n_states, n_actions))

    def get_best_action(self, state):
        return np.argmax(self.qvalues[state])

    def get_action(self, state):
        r = np.random.choice([0, 1], p=[self.epsilon, 1 - self.epsilon])
        if r == 0:
            a = np.random.choice(n_actions)
        else:
            a = self.get_best_action(state)

        return a

    def update(self, state, reward, next_state, action):
        q_max = np.max(self.qvalues[next_state])
        q_estimated = reward + self.gamma * q_max
        self.qvalues[state, action] = self.alpha * q_estimated + \
                                      (1 - self.alpha) * q_max


if __name__ == '__main__':
    env = gym.make('Taxi-v3')
    env.reset()
    env.render()

    n_states = env.observation_space.n 
    n_actions = env.action_space.n

    print(f'States number: {n_states}, Actions number: {n_actions}')

    s_new, reward, is_done, _ = env.step(0)

    alpha = 0.5
    epsilon = 0.4
    gamma = 0.9
    n_iterations = 10000
    max_iterations = 10000

    agent = QLearningAgent(alpha, epsilon, gamma, n_actions, n_states)
    max_reward = 0
    for i in range(n_iterations):
        s = env.reset()
        total_reward = 0.0
        for _ in range(max_iterations):
            a = agent.get_action(s)
            s_new, r, is_done, _ = env.step(a)

            agent.update(s, reward, s_new, a)
            total_reward += r
            s = s_new
            if is_done:
                break
        if max_reward < total_reward:
            max_reward = total_reward
        if i % 100 == 0:
            agent.epsilon = (agent.epsilon - 0.0025) if agent.epsilon - 0.0025 > 0 else 0
            print(f'Iteration: {i}, Total reward: {total_reward}, Max reward: {max_reward},  Epsilon: {agent.epsilon}')

    s = env.reset()
    agent.epsilon = 0
    env.render()
    for _ in range(max_iterations):
        a = agent.get_action(s)
        s_new, r, is_done, _ = env.step(a)
        if is_done:
            break
        s = s_new
        env.render()
