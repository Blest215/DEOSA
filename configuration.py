import tensorflow as tf
import numpy as np
import json
import os
import errno

from models.observation import Observation
from reinforcement_learning.reward import RewardFunction


class Configuration:
    """ Configuration: class contains configurations for experiments """
    def __init__(self,
                 # environmental configurations
                 num_user, num_service, width, height, depth, observation,
                 user_constructor, service_constructor,
                 max_speed,
                 reward_function,

                 # learning configurations
                 num_episode, num_step,
                 memory_size, batch_size, learning_rate, discount_factor,
                 eps_init, eps_final, eps_decay,
                 agent,

                 datetime,
                 summary_path):
        """ __init__: simply add all the given parameters as its attributes """
        # Environment
        self.num_user = num_user
        self.num_service = num_service
        self.width = width
        self.height = height
        self.depth = depth

        # Observation
        self.observation = observation

        # Dynamics
        self.max_speed = max_speed

        # Constructors
        self.user_constructor = user_constructor
        self.service_constructor = service_constructor

        # Reward
        self.reward_function = reward_function

        # Experiment
        self.num_episode = num_episode
        self.num_step = num_step

        # Learning
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        # Epsilon-greedy policy
        self.eps_init = eps_init
        self.eps_final = eps_final
        self.eps_decay = eps_decay

        self.agent = agent

        # Summary
        self.datetime = datetime
        self.summary_path = summary_path

    def save(self):
        class CustomEncoder(json.JSONEncoder):
            def default(self, obj):
                try:
                    return json.JSONEncoder.default(self, obj)
                except TypeError:
                    return str(obj)


        file_path = self.summary_path
        if not os.path.exists(os.path.dirname(file_path)):
            try:
                os.makedirs(os.path.dirname(file_path))
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
        with open(file_path, 'w') as f:
            f.write(json.dumps(self.__dict__, indent=4, cls=CustomEncoder))
            f.close()
