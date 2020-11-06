import argparse
import datetime
import tensorflow as tf

from experiment import Experiment
from models.environment.environment import Environment
from reinforcement_learning.reward import RewardFunction
from reinforcement_learning.agent.baselines import RandomSelectionAgent, NearestSelectionAgent, \
    NoHandoverSelectionAgent, GreedySelectionAgent
from reinforcement_learning.agent.deosa import DEOSA

from models.environment.observation import EuclideanObservation, FullObservation
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
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    env = Environment(num_user=1,
                      num_service=100,
                      width=100,  # x
                      height=20,  # y
                      depth=3,  # z
                      max_speed=1,
                      # observation=EuclideanObservation(observation_range=10),
                      observation=FullObservation(),
                      reward_function=RewardFunction(effectiveness_function=VisualEffectivenessFunction(
                          visual_field_max=80,
                          viewing_angle_max=70
                      ), penalty=0.5, weight=None))

    experiment = Experiment(env=env,
                            num_episode=1000,
                            num_step=100,
                            now=now)

    if args.agent == "random" or args.agent == "all":
        experiment.run(agent=RandomSelectionAgent(env, now), train=False)

    if args.agent == "greedy" or args.agent == "all":
        experiment.run(agent=GreedySelectionAgent(env, now), train=False)

    if args.agent == "nearest" or args.agent == "all":
        experiment.run(agent=NearestSelectionAgent(env, now), train=False)

    if args.agent == "nohandover" or args.agent == "all":
        experiment.run(agent=NoHandoverSelectionAgent(env, now), train=False)

    if args.agent == "DEOSA" or args.agent == "all":
        experiment.run(
            agent=DEOSA(env, now,
                        memory_size=1000,
                        batch_size=10,
                        learning_rate=4e-7,
                        discount_factor=.99,
                        tau=0.001,
                        hidden_units=[512, 512, 512, 512, 512, 512, 512, 512, 512, 512],
                        activation='relu',
                        eps_init=1.0,
                        eps_final=0.1,
                        # set decaying rate according to the number of episodes:
                        # to make epsilon reaches eps_final at the end
                        # eps_decay=np.power(1e-2/1.0, 1 / 1000),
                        eps_decay=0.99),
            train=True
        )


if __name__ == '__main__':
    main()
