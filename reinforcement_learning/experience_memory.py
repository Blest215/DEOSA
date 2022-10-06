import random

import numpy as np
import operator
from abc import abstractmethod


class ExperienceMemory:
    def __init__(self, size):
        self.memory = []
        self.size = size

        self.__setting__ = self.__dict__.copy()
        self.__setting__["name"] = type(self).__name__

    @abstractmethod
    def add(self, observation, action, reward, next_observation, done):
        self.memory.append([observation, action, reward, next_observation, done])

    def sample(self, batch_size):
        if 0 < len(self.memory) < batch_size:
            return None
        # observations, actions, rewards, next_observations, done
        return map(np.asarray, zip(*random.sample(self.memory, batch_size)))

    def is_full(self):
        return len(self.memory) == self.size

    def is_empty(self):
        return len(self.memory) == 0


# Basic experience memory that randomly sampling experiences for DQN
class BasicExperienceMemory(ExperienceMemory):
    def add(self, observation, action, reward, next_observation, done):
        while self.is_full():
            # Random pop-up
            # self.memory.pop(random.randrange(0, len(self.memory)))
            # FIFO
            self.memory.pop(0)
        self.memory.append([observation, action, reward, next_observation, done])


# Balancing experience memory that balancing reward distribution
class BalancingExperienceMemory(ExperienceMemory):
    def __init__(self, size):
        self.count = {}

        super().__init__(size)

    def add(self, observation, action, reward, next_observation, done):
        while self.is_full():
            """ pop a memory instance from largest reword set, balancing reward distribution """
            target_reward = max(self.count.items(), key=operator.itemgetter(1))[0]
            for m in self.memory:
                # FIFO
                if m["reward"].get_overall_score() == target_reward:
                    self.count[target_reward] -= 1
                    self.memory.remove(m)
                    break

        # TODO only works for discrete value of reward
        reward_value = reward.get_overall_score()
        if reward_value not in self.count:
            self.count[reward_value] = 1
        else:
            self.count[reward_value] += 1

        self.memory.append([observation, action, reward, next_observation, done])
