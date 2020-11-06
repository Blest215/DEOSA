import tensorflow as tf
import numpy as np
import inspect

from reinforcement_learning.agent.agent import Agent
from reinforcement_learning.experience_memory import BasicExperienceMemory
from reinforcement_learning.network.deosa_network import DEOSANetwork
from utils import get_summary_path


class DEOSA(Agent):
    def __init__(self, env, now,
                 memory_size, batch_size, learning_rate, discount_factor, tau,
                 hidden_units, activation,
                 eps_init, eps_final, eps_decay):

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.hidden_units = hidden_units
        self.activation = activation
        self.tau = tau

        """ Network settings """
        self.main_network = DEOSANetwork(learning_rate=learning_rate, discount_factor=discount_factor,
                                         hidden_units=hidden_units, activation=activation)
        self.target_network = DEOSANetwork(learning_rate=learning_rate, discount_factor=discount_factor,
                                           hidden_units=hidden_units, activation=activation)
        self.main_network.set_target_network(self.target_network)

        """ Experience memory settings """
        self.memory = BasicExperienceMemory(memory_size)
        self.batch_size = batch_size

        """ Epsilon greedy settings """
        self.eps = eps_init
        self.eps_init = eps_init
        self.eps_final = eps_final
        self.eps_decay = eps_decay

        self.selection_code = inspect.getsource(self.selection)
        self.normalization_code = inspect.getsource(self.convert_observations)

        Agent.__init__(self, env, now)

        with open(get_summary_path(agent=self.name, datetime=self.now, filename="model.txt"), 'w') as f:
            self.main_network.summary(print_fn=lambda x: f.write(x + '\n'))

    def selection(self, user, services):
        if self.train and np.random.random() < self.eps:
            """ epsilon-greedy """
            selection = np.random.choice(range(len(services)))
        else:
            """ calculate Q-value for each service (action) """
            Q_set = self.main_network([self.convert_observations(user, services)])
            # when effectiveness is available
            # effectiveness = [self.env.reward_function.measure(user, service) for service in services]
            # max_effectiveness = max(effectiveness)
            # mask = tf.constant([-10000 if effectiveness[i] < max_effectiveness else 0 for i in range(len(services))], dtype=tf.float64)
            # Q_set = tf.add(Q_set, mask)
            selection = np.argmax(Q_set)

        return services[selection], selection

    def learn(self, observation, action_index, reward, next_observation, done):
        """ learn: performs learning process and updates the parameters of the network """
        self.memory.add(observation=self.convert_observations(observation["user"], observation["services"]),
                        action=action_index,
                        reward=float(reward),
                        next_observation=self.convert_observations(next_observation["user"], next_observation["services"]),
                        done=done)
        loss_list = []

        if self.memory.is_full():
            """ perform learning process of the network if the memory is full of experiences """
            observations, actions, rewards, next_observations, done = self.memory.sample(self.batch_size)

            loss_list.append(self.main_network.update(observation=observations,
                                                      action=actions,
                                                      reward=rewards,
                                                      next_observation=next_observations,
                                                      done=done))

            """ update target network """
            self.main_network.update_target_network(self.tau)

            return np.mean(loss_list)

    def pre_episode_process(self, *kwargs):
        """ pre_episode_process """

        """ copy target network """
        # self.main_network.copy_from_target_network()

    def post_episode_process(self, i_episode):
        """ post_episode_process """

        """ epsilon decaying for each episode """
        if self.eps > self.eps_final:
            self.eps = self.eps_decay * self.eps
        else:
            self.eps = self.eps_final

        if i_episode % 10 == 9:
            """ save the trained model for each 10-episodes """
            # self.main_network.save(get_summary_path(self.name, self.datetime, "train"), save_format='tf')
            # tf.saved_model.save(self.main_network, get_summary_path(self.name, self.datetime, "train"))

    def convert_observations(self, user, services):
        """ convert_observations: converts user and services information into matrix for the TensorFlow network """
        num_service = len(services)
        user_tile = np.tile(user.vectorize(), (num_service, 1))
        service_tile = np.array(
            [service.vectorize() for service in services]
        )
        observations = np.concatenate((user_tile, service_tile), axis=1)

        return observations
