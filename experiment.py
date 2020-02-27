from models.environment import Environment
from reinforcement_learning.agent import *


class Experiment:
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def run(self):
        pass


class EffectDrivenMediumSelectionExperiment(Experiment):
    """
        EffectDrivenMediumSelectionExperiment: Effect-driven and dynamic selection of physical media
    """
    def __init__(self, configuration):
        self.configuration = configuration

        self.env = Environment(num_user=configuration.num_user,
                               num_service=configuration.num_service,
                               width=configuration.width,
                               height=configuration.height,
                               depth=configuration.depth,
                               observation=configuration.observation,
                               user_constructor=configuration.user_constructor,
                               service_constructor=configuration.service_constructor,
                               reward_function=configuration.reward_function)

        self.num_episode = configuration.num_episode
        self.num_step = configuration.num_step
        self.memory_size = configuration.memory_size
        self.batch_size = configuration.batch_size

        self.datetime = configuration.datetime

        """ In the code, only one agent should be constructed, Otherwise, error occurs in summary """
        if configuration.agent == "random":
            self.agent = RandomSelectionAgent("Random", self.env, self.datetime, self.num_episode, self.num_step)
        if configuration.agent == "nearest":
            self.agent = NearestSelectionAgent("Nearest", self.env, self.datetime, self.num_episode, self.num_step)
        if configuration.agent == "nohandover":
            self.agent = NoHandoverSelectionAgent("NoHandover", self.env, self.datetime, self.num_episode, self.num_step)
        if configuration.agent == "greedy":
            self.agent = GreedySelectionAgent("Greedy", self.env, self.datetime, self.num_episode, self.num_step)
        if configuration.agent == "EDMS(DQN)":
            self.agent = EDMSAgentDQN("EDMS(DQN)", self.env, self.datetime, self.num_episode, self.num_step,
                                      memory_size=self.memory_size,
                                      batch_size=self.batch_size,
                                      learning_rate=configuration.learning_rate,
                                      discount_factor=configuration.discount_factor,
                                      eps_init=configuration.eps_init,
                                      eps_final=configuration.eps_final,
                                      eps_decay=configuration.eps_decay)

    def reset(self):
        self.env.reset()

    def run(self):
        self.configuration.save()
        with tf.device('/GPU:1'):
            self.agent.run("train")
