import tensorflow as tf
import numpy as np
import time
from abc import abstractmethod

from models.environment.environment import Environment
from utils import variable_summaries, get_summary_path


class Agent:
    def __init__(self, name, env, now):
        self.name = name

        self.__setting__ = self.__dict__.copy()

        self.train = False

        assert isinstance(env, Environment)
        self.env = env
        self.now = now

    @abstractmethod
    def selection(self, user, services):
        """ return selected service object and its index """
        return None, 0

    def pre_episode_process(self, *kwargs):
        """ pre_episode_process: a function called before each episode """
        pass

    def post_episode_process(self, *kwargs):
        """ post_episode_process: a function called after each episode """
        pass

    def learn(self, observation, action_index, reward, next_observation, done):
        pass

    def run(self, num_episode, num_step, train=False):
        """ run: run train or test simulations according to the given mode """

        """ summary writer of TensorFlow """
        writer = tf.summary.create_file_writer(get_summary_path(self.name, self.now, "train" if train else "test"))
        print("Train phase" if train else "Test phase")
        self.train = train

        for i_episode in range(num_episode):
            tf.summary.trace_on()
            with writer.as_default():
                print("Episode %d" % i_episode)

                """ Set-up """
                tf.summary.experimental.set_step(i_episode)
                # np.random.seed(i_episode + (0 if train else num_episode) + 1000)
                self.pre_episode_process(i_episode)

                loss_list = []  # only for training mode
                reward_list = []
                execution_time_list = []
                observation = self.env.reset()

                episode_start = time.time()

                """ 
                since service selection is non-episodic task, 
                restrict maximum step rather than observe done-signal 
                """
                for i_step in range(num_step):
                    start_time = time.time()
                    """ select action """
                    action, action_index = self.selection(observation["user"], observation["services"])
                    execution_time_list.append(time.time() - start_time)

                    """ perform the selected action on the environment """
                    next_observation, reward, done = self.env.step(action)
                    """ add reward to total score """
                    reward_list.append(reward)

                    if done:
                        """ skip training if done """
                        break

                    if train:
                        """ perform learning process if the mode is train """
                        loss = self.learn(observation, action_index, reward, next_observation, done)
                        if loss:
                            loss_list.append(loss)

                    """ set observation to next state """
                    observation = next_observation

                self.post_episode_process(i_episode)

                """ summaries """
                variable_summaries('execution_time', execution_time_list, step=i_episode)
                variable_summaries('reward', [float(reward) for reward in reward_list], step=i_episode)
                variable_summaries('effectiveness', [reward.effectiveness for reward in reward_list], step=i_episode)
                variable_summaries('penalty', [reward.penalty for reward in reward_list], step=i_episode)
                print("Episode {i} ends with average reward {reward} ({time:.3f} seconds)".format(
                    i=i_episode,
                    reward=np.mean(reward_list),
                    time=time.time() - episode_start)
                )
                if train and loss_list:
                    variable_summaries('loss', loss_list, step=i_episode)
                    print("Average loss was: {loss}".format(loss=np.mean(loss_list)))

                tf.summary.trace_export(name="EDMS agent (DQN version) experiment")
