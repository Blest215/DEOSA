import tensorflow as tf
import numpy as np
import time
from abc import abstractmethod

from models.environment.environment import Environment
from utils import variable_summaries, get_summary_path


class Agent:
    def __init__(self, name, env, datetime, num_episode, num_step):
        self.name = name

        assert isinstance(env, Environment)
        self.env = env

        """ training configuration """
        self.num_episode = num_episode
        self.num_step = num_step

        """ date of now, for logging """
        self.datetime = datetime

        """ whether performing training or not """
        self.train = False

        """ summary writers of TensorFlow """
        self.train_summary_writer = tf.summary.create_file_writer(
            get_summary_path(self.name, self.datetime, "train"))
        self.test_summary_writer = tf.summary.create_file_writer(
            get_summary_path(self.name, self.datetime, "test"))

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

        """ set training or testing mode according to command """
        assert mode == "train" or mode == "test"
        if mode == "train":
            self.train = True
            writer = self.train_summary_writer
        else:
            self.train = False
            writer = self.test_summary_writer
        print("{mode} phase".format(mode=mode))

        tf.summary.trace_on()
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

                    if self.train:
                        """ perform learning process if the mode is train """
                        loss = self.learn(observation, action_index, reward, next_observation, done)
                        if loss:
                            loss_list.append(loss)

                    if done:
                        break

                    """ set observation to next state """
                    observation = next_observation

                self.post_episode_process(i_episode)

                """ summaries """
                variable_summaries('execution_time', execution_time_list, step=i_episode)
                variable_summaries('reward', reward_list, step=i_episode)
                print("Episode {i} ends with average reward {reward}".format(i=i_episode,
                                                                             reward=np.mean(reward_list)))
                if self.train and loss_list:
                    variable_summaries('loss', loss_list, step=i_episode)
                    print("Average loss was: {loss}".format(loss=np.mean(loss_list)))

            tf.summary.trace_export()
