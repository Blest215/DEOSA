import tensorflow as tf
import numpy as np
import random
import time
from abc import abstractmethod

from reinforcement_learning.network import EDMSNetworkDQN, EDMSNetworkDQN
from reinforcement_learning.experience_memory import BasicExperienceMemory, BalancingExperienceMemory
from models.environment import Environment
from utils import variable_summaries


class Agent:
    def __init__(self, name, env, datetime, num_episode, num_step):
        self.name = name

        assert isinstance(env, Environment)
        self.env = env

        # Training configuration
        self.num_episode = num_episode
        self.num_step = num_step

        # Date of now, for logging
        self.datetime = datetime

        self.train_summary_writer = tf.summary.create_file_writer(
            './summary/{name}/{date}/train'.format(name=self.name, date=self.datetime))
        self.test_summary_writer = tf.summary.create_file_writer(
            './summary/{name}/{date}/test'.format(name=self.name, date=self.datetime))

    @abstractmethod
    def selection(self, user, services):
        """ return selected service object and its index """
        return None, 0

    def post_episode_process(self, *kwargs):
        """ post_episode_process: a function called after each episode """
        pass

    def learn(self, observation, action_index, reward, next_observation, done):
        pass

    def run(self, mode="test"):
        """ run: run train or test simulations according to the given mode """

        assert mode == "train" or mode == "test"
        if mode == "train":
            writer = self.train_summary_writer
        else:
            writer = self.test_summary_writer
        print("{mode} phase".format(mode=mode))

        with writer.as_default():
            for i_episode in range(self.num_episode):
                print("Episode %d" % i_episode)
                # random.seed(i_episode)

                loss_list = []  # only for training mode
                reward_list = []
                execution_time_list = []
                observation = self.env.reset()

                """ 
                since service selection is non-episodic task, 
                restrict maximum step rather than observe done-signal 
                """
                for i_step in range(self.num_step):
                    start_time = time.time()
                    """ select action """
                    action, action_index = self.selection(observation["user"], observation["services"])
                    execution_time_list.append(time.time() - start_time)

                    """ perform the selected action on the environment """
                    next_observation, reward, done = self.env.step(action)
                    """ add reward to total score """
                    reward_list.append(reward.get_overall_score())

                    if mode == "train":
                        """ perform learning process if the mode is train """
                        loss = self.learn(observation, action_index, reward, next_observation, done)
                        if loss:
                            loss_list.append(loss)

                    if done:
                        break

                    """ set observation to next state """
                    observation = next_observation

                """ summaries """
                variable_summaries('execution_time', execution_time_list, step=i_episode)
                variable_summaries('reward', reward_list, step=i_episode)
                print("Episode {i} ends with average reward {reward}".format(i=i_episode,
                                                                             reward=np.mean(reward_list)))
                if mode == "train" and loss_list:
                    variable_summaries('loss', loss_list, step=i_episode)
                    print("Average loss was: {loss}".format(loss=np.mean(loss_list)))


class RandomSelectionAgent(Agent):
    """ RandomSelectionAgent: a baseline agent that selects services randomly """

    def selection(self, user, services):
        index = random.choice(range(len(services)))
        return services[index], index


class NearestSelectionAgent(Agent):
    """ ClosestSelectionAgent: a baseline agent that selects the nearest service"""

    def selection(self, user, services):
        minimum = 1000000
        index = -1
        for i in range(len(services)):
            if user.get_distance(services[i].device) < minimum:
                index = i
                minimum = user.get_distance(services[i].device)
        return services[index], index


class NoHandoverSelectionAgent(Agent):
    """ NoHandoverSelectionAgent: a baseline agent that minimizes the number of handovers """

    def selection(self, user, services):
        for i in range(len(services)):
            if services[i].in_use and services[i].user == user:
                return services[i], i
        """ if no service is currently in-use, select randomly """
        index = random.choice(range(len(services)))
        return services[index], index


class GreedySelectionAgent(Agent):
    """ GreedySelectionAgent: a baseline agent that selects best one currently, in terms of effectiveness """

    def selection(self, user, services):
        maximum = -1000000
        index = -1
        for i in range(len(services)):
            reward = self.env.reward_function.measure(user, services[i])
            if reward.effectiveness > maximum:
                index = i
                maximum = reward.effectiveness
        return services[index], index


class EDMSAgentDQN(Agent):
    def __init__(self, name, env, date, num_episode, num_step,
                 memory_size, batch_size, learning_rate, discount_factor,
                 eps_init, eps_final, eps_decay):
        Agent.__init__(self, name, env, date, num_episode, num_step)

        self.main_network = EDMSNetworkDQN(name="main",
                                           learning_rate=learning_rate,
                                           discount_factor=discount_factor,
                                           observation_size=self.env.get_observation_size(),
                                           action_size=self.env.get_action_size())

        """ Experience memory setting """
        self.memory = BasicExperienceMemory(memory_size)
        self.batch_size = batch_size

        """ Epsilon greedy setting """
        self.eps_init = eps_init
        self.eps_final = eps_final
        self.eps_decay = eps_decay

    def selection(self, user, services):
        Q_set = self.main_network.sample(sess, user.vectorize(), [service.vectorize() for service in services])
        selection = np.argmax(Q_set)
        return services[selection], selection

    def train(self):
        print("Train phase")

        writer = tf.summary.FileWriter('{path}/{name}/{date}/train'.format(path=tf.flags.FLAGS.summary_path,
                                                                           name=self.name, date=self.datetime),
                                       sess.graph)

        """ Epsilon greedy configuration """
        eps = self.eps_init

        for i_episode in range(self.num_episode):
            print("Episode %d" % i_episode)

            reward_list = []
            loss_list = []
            execution_time_list = []
            observation = self.env.reset()

            for i_step in range(self.num_step):
                start_time = time.time()
                """ select action: epsilon-greedy """
                if random.random() <= eps:
                    action_index = random.choice(range(len(observation["services"])))
                    action = observation["services"][action_index]
                else:
                    action, action_index = self.selection(sess, observation["user"], observation["services"])
                execution_time_list.append(time.time() - start_time)

                """ perform the selected action on the environment """
                next_observation, reward, done = self.env.step(action)
                reward_list.append(reward)

                self.memory.add(observation, action_index, reward, next_observation, done)

                if self.memory.is_full():
                    """ training the network """
                    loss_list += self.learn(sess)

                if done:
                    break

                """ set observation to next state """
                observation = next_observation

            """ epsilon decaying for each episode """
            if eps > self.eps_final:
                eps = self.eps_decay * eps
            else:
                eps = self.eps_final

            if loss_list:
                self.summarize_episode(sess, writer, i_episode, loss_list, reward_list, execution_time_list)
                print("Episode {i} ends with average score {reward}, loss {loss}".format(i=i_episode,
                                                                                         reward=np.mean(reward_list),
                                                                                         loss=np.mean(loss_list)))
            else:
                self.summarize_episode(sess, writer, i_episode, [None], reward_list, execution_time_list)
                print("Episode {i} ends with average score {reward}".format(i=i_episode,
                                                                            reward=np.mean(reward_list)))

            if isinstance(self.memory, BalancingExperienceMemory):
                print("Reward distribution: %s" % self.memory.count)

    def learn(self):
        batch = self.memory.sample(self.batch_size)
        loss_list = []

        for memory in batch:
            loss, _ = self.main_network.update(sess=sess,
                                               observation=memory["observation"]["user"].vectorize(),
                                               actions=[service.vectorize() for service in
                                                        memory["observation"]["services"]],
                                               action=memory["action"],
                                               reward=memory["reward"].get_overall_score(),
                                               next_observation=memory["next_observation"]["user"].vectorize(),
                                               next_actions=[service.vectorize() for service in
                                                             memory["next_observation"]["services"]],
                                               done=memory["done"])
            loss_list.append(loss)
        return loss_list


class NewEDMSAgentDQN(Agent):
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
            observation = self.convert_observations(user, services)
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
                                                          reward=memory["reward"],
                                                          next_observation=memory["next_observation"],
                                                          done=memory["done"]))
            return np.mean(loss_list)

    def post_episode_process(self, *kwargs):
        """ epsilon decaying for each episode """
        if self.eps > self.eps_final:
            self.eps = self.eps_decay * self.eps
        else:
            self.eps = self.eps_final

    @staticmethod
    def convert_observations(user, services):
        """ convert_observations: convert user and services information into matrix for the TensorFlow network """
        num_service = len(services)
        user_tile = np.tile(user.vectorize(), (num_service, 1))
        service_tile = [service.vectorize() for service in services]
        observation = np.concatenate((user_tile, service_tile), axis=1)
        return observation
