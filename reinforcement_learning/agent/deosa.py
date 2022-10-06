import tensorflow as tf
import numpy as np
import inspect

from reinforcement_learning.agent.agent import Agent
from reinforcement_learning.experience_memory import BasicExperienceMemory
from utils import get_summary_path


class DEOSA(Agent):
    def __init__(self, env, now,
                 memory_size, batch_size,
                 network,
                 eps_init, eps_final, eps_decay):

        """ Network settings """
        self.main_network = network
        self.main_network.create_target_network()

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

        Agent.__init__(self, "{agent}/{network}".format(agent=type(self).__name__,
                                                        network=type(network).__name__),
                       env, now)

        with open(get_summary_path(agent=self.name, datetime=self.now, filename="model.txt"), 'w') as f:
            self.main_network.summary(print_fn=lambda x: f.write(x + '\n'))

    def selection(self, user, services):
        selection = self.main_network.selection(
            np.squeeze(self.main_network([self.convert_observations(user, services)])),
            self.eps if self.train else 0
        )

        return services[selection], selection

    def learn(self, observation, action_index, reward, next_observation, done):
        """ learn: performs learning process and updates the parameters of the network """
        self.memory.add(observation=self.convert_observations(observation["user"], observation["services"]),
                        action=action_index,
                        reward=float(reward),
                        next_observation=self.convert_observations(next_observation["user"],
                                                                   next_observation["services"]),
                        done=done)
        loss_list = []

        if self.memory.is_full():
            """ perform learning process of the network if the memory is full of experiences """
            observations, actions, rewards, next_observations, done = self.memory.sample(self.batch_size)

            loss_list.append(self.main_network.update(observations=observations,
                                                      actions=actions,
                                                      rewards=rewards,
                                                      next_observations=next_observations,
                                                      done=done))

            """ update target network """
            self.main_network.update_target_network()

            return np.mean(loss_list)

    def pre_episode_process(self, *kwargs):
        """ pre_episode_process """

        """ copy target network """
        self.main_network.copy_from_target_network()

    def post_episode_process(self, i_episode):
        """ post_episode_process """

        """ epsilon decaying for each episode """
        if self.eps > self.eps_final:
            self.eps = self.eps_decay * self.eps
        else:
            self.eps = self.eps_final

        # if i_episode % 10 == 9:
        #     """ save the trained model for each 10-episodes """
        #     self.main_network.save(get_summary_path(self.name, self.datetime, "train"), save_format='tf')
        #     tf.saved_model.save(self.main_network, get_summary_path(self.name, self.datetime, "train"))

    def convert_observations(self, user, services):
        """ convert_observations: converts user and services information into matrix for the TensorFlow network """

        num_service = len(services)
        user_tile = np.tile(user.vectorize(), (num_service, 1))
        service_tile = np.array(
            [service.vectorize() for service in services]
        )
        observations = np.concatenate((user_tile, service_tile), axis=1)

        return observations
