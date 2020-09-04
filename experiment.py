import tensorflow as tf
import json
import os
import errno

from utils import get_summary_path


class Experiment:
    """
        Experiment: The experiment class for the evaluation of dynamic and effect-driven selection of output-services
    """
    def __init__(self, env, agent, now, num_episode, num_step):
        """ __init__: simply add all the given parameters as its attributes """

        self.env = env

        self.agent = agent

        # Iteration settings
        self.num_episode = num_episode
        self.num_step = num_step

        # Experiment datetime
        self.now = now

    def reset(self):
        self.env.reset()

    def run(self):
        self.save()

        self.env.reset()
        with tf.device('GPU:1'):
            self.agent.run(num_episode=self.num_episode,
                           num_step=self.num_step,
                           mode="train")

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
        if not os.path.exists(os.path.dirname(file_path)):
            try:
                os.makedirs(os.path.dirname(file_path))
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
        with open(file_path, 'w') as f:
            f.write(json.dumps(self.__dict__, indent=4, cls=CustomJSONEncoder))
            f.close()
