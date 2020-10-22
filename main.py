import argparse
import datetime
import os
import tensorflow as tf

from experiment import Experiment
from models.environment.environment import Environment
from reinforcement_learning.reward import RewardFunction
from reinforcement_learning.agent.baselines import RandomSelectionAgent, NearestSelectionAgent, \
    NoHandoverSelectionAgent, GreedySelectionAgent
from reinforcement_learning.agent.deosa import DEOSA

from models.environment.observation import EuclideanObservation
from models.effectiveness import VisualEffectivenessFunction

parser = argparse.ArgumentParser()
parser.add_argument("agent", help="the name of agent to simulate")
args = parser.parse_args()


tf.keras.backend.set_floatx('float64')


def main():
    """
        Environment settings
            - single user
            - single service selected
            - one-to-one matching between services and devices
            - partial observation based on Euclidean-distance
            - 3-dimensional space
    """
    width = 200  # x
    height = 20  # y
    depth = 3  # z
    max_speed = 1
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    env = Environment(num_user=1,
                      num_service=100,
                      width=width,
                      height=height,
                      depth=depth,
                      max_speed=max_speed,
                      observation=EuclideanObservation(observation_range=10),
                      reward_function=RewardFunction(effectiveness_function=VisualEffectivenessFunction(
                          visual_field_max=80,
                          viewing_angle_max=70
                      ), weight=None))

    """ In the code, only one agent should be constructed, Otherwise, error occurs in summary """
    agent = None
    if args.agent == "random":
        agent = RandomSelectionAgent(env, now)
    if args.agent == "nearest":
        agent = NearestSelectionAgent(env, now)
    if args.agent == "nohandover":
        agent = NoHandoverSelectionAgent(env, now)
    if args.agent == "greedy":
        agent = GreedySelectionAgent(env, now)
    if args.agent == "DEOSA":
        agent = DEOSA(env, now,
                      memory_size=1000,
                      batch_size=100,
                      learning_rate=1e-11,
                      discount_factor=.95,
                      hidden_units=[512, 512, 512, 512, 512, 512, 512],
                      activation='relu',
                      eps_init=1.0,
                      eps_final=1e-2,
                      # set decaying rate according to the number of episodes:
                      # to make epsilon reaches eps_final at the end
                      # eps_decay=np.power(1e-2/1.0, 1 / 1000),
                      eps_decay=0.99)

    experiment = Experiment(env=env,
                            agent=agent,
                            num_episode=100,
                            num_step=100,
                            now=now)

    experiment.reset()
    experiment.run()


if __name__ == '__main__':
    main()
