from collections import deque
import random

# Discussed with groupemates and taken from
# The idea on slides: Lection 3
class Experience:
    def __init__(self):
        self.SIZE = 10000
        self.BATCH = 600
        self.data = deque(maxlen=self.SIZE)

    def add(self, states, actions, new_states, rewards, dones):
        episodes = list(zip(states, actions, new_states, rewards, dones))
        self.data.extend(episodes)

    def replay(self):
        while len(self.data) > self.SIZE:
            self.data.popleft()
        batch = random.sample(self.data, min(self.BATCH, len(self.data)))
        state, action, next_state, reward, terminal = zip(*batch)
        return state, action, next_state, reward, terminal
