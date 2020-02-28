import random

import numpy as np

from reinforcement_learning.agent.agent import Agent
from reinforcement_learning.experience_memory import BasicExperienceMemory
from reinforcement_learning.network.dqn import EDMSNetworkDQN


class EDMSAgentDQN(Agent):
    def __init__(self, name, env, date, num_episode, num_step,
                 memory_size, batch_size, learning_rate, discount_factor,
                 eps_init, eps_final, eps_decay):
        Agent.__init__(self, name, env, date, num_episode, num_step)

        self.main_network = EDMSNetworkDQN(observation_size=self.env.get_observation_size(),
                                           learning_rate=learning_rate,
                                           discount_factor=discount_factor,
                                           hidden_units=[128, 128],
                                           activation='relu')

        """ Experience memory setting """
        self.memory = BasicExperienceMemory(memory_size)
        self.batch_size = batch_size

        """ Epsilon greedy setting """
        self.eps = eps_init
        self.eps_init = eps_init
        self.eps_final = eps_final
        self.eps_decay = eps_decay

    def selection(self, user, services):
        if np.random.random() < self.eps:
            """ epsilon-greedy """
            selection = random.choice(range(len(services)))
        else:
            observation = self.normalize_observations(self.convert_observations(user, services))
            """ calculate Q-value for each service (action) """
            Q_set = self.main_network(observation)
            selection = np.argmax(Q_set)

        return services[selection], selection

    def learn(self, observation, action_index, reward, next_observation, done):
        """ learn: performs learning process and updates the parameters of the network """
        self.memory.add(observation=self.convert_observations(observation["user"], observation["services"]),
                        action=action_index,
                        reward=reward,
                        next_observation=self.convert_observations(next_observation["user"], observation["services"]),
                        done=done)
        loss_list = []

        if self.memory.is_full():
            """ perform learning process of the network if the memory is full of experiences """
            batch = self.memory.sample(self.batch_size)

            for memory in batch:
                loss_list.append(self.main_network.update(observation=memory["observation"],
                                                          action=memory["action"],
                                                          reward=float(memory["reward"]),
                                                          next_observation=memory["next_observation"],
                                                          done=memory["done"]))
            return np.mean(loss_list)

    def post_episode_process(self, i_episode):
        """ post_episode_process: overriding post_episode_process method """

        if self.train:
            """ epsilon decaying for each episode """
            if self.eps > self.eps_final:
                self.eps = self.eps_decay * self.eps
            else:
                self.eps = self.eps_final

            if i_episode % 10 == 9:
                """ save the trained model for each 10-episodes """
                # self.main_network.save(get_summary_path(self.name, self.datetime, "train"), save_format='tf')
                # tf.saved_model.save(self.main_network, get_summary_path(self.name, self.datetime, "train"))

    @staticmethod
    def normalize_observations(observations):
        """ normalize_observations: normalizes the observation values by taking logarithm after adding 1 """
        return np.log(np.add(observations, 1))

    @staticmethod
    def convert_observations(user, services):
        """ convert_observations: converts user and services information into matrix for the TensorFlow network """
        num_service = len(services)
        user_tile = np.tile(user.vectorize(), (num_service, 1))
        service_tile = np.array([service.vectorize() for service in services])
        observation = np.concatenate((user_tile, service_tile), axis=1)
        return observation