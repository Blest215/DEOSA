import tensorflow as tf
import json
import os

from reinforcement_learning.agent.agent import Agent
from reinforcement_learning.agent.deosa import DEOSA
from utils import get_summary_path


class Experiment:
    """
        Experiment: The experiment class for the evaluation of dynamic and effect-driven selection of output-services
    """
    def __init__(self, now, num_episode, num_step):
        """ __init__: simply add all the given parameters as its attributes """

        self.agent = None

        # Iteration settings
        self.num_episode = num_episode
        self.num_step = num_step

        # Experiment datetime
        self.now = now

    def run(self, agent, train=False):
        assert isinstance(agent, Agent)
        self.agent = agent
        self.save()

        # with tf.device('GPU:1'):
        if train:
            assert isinstance(self.agent, DEOSA)
            self.agent.run(num_episode=self.num_episode,
                           num_step=self.num_step,
                           train=True)
        self.agent.run(num_episode=self.num_episode,
                       num_step=self.num_step,
                       train=False)

    def save(self):
        """
        save: save the configurations of the experiment
        """

        class CustomJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                try:
                    return json.JSONEncoder.default(self, obj)
                except TypeError:
                    try:
                        return obj.__setting__
                    except (TypeError, AttributeError):
                        return str(obj)

        file_path = get_summary_path(agent=self.agent.name, datetime=self.now, filename="configuration.txt")
        with open(file_path, 'w') as f:
            f.write(json.dumps(self.__dict__, indent=4, cls=CustomJSONEncoder))
            f.close()
