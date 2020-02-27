import tensorflow as tf
import argparse
import datetime

from configuration import Configuration
from experiment import EffectDrivenMediumSelectionExperiment
from reinforcement_learning.reward import HandoverPenaltyRewardFunction

from models.observation import EuclideanObservation, FullObservation
from models.effectiveness import VisualEffectiveness
from models.entity import User, Service, DisplayDevice
from models.physics import *
from models.constructor import UserConstructor, VisualServiceConstructor


""" format of path that collects summary for each experiment """
SUMMARY_PATH = "./summary/{agent}/{date}/{file}"


def get_summary_path(agent, date, filename):
    """ get_summary_path: returns a file path for collecting summary for each experiment """
    return SUMMARY_PATH.format(agent=agent, date=date, file=filename)


parser = argparse.ArgumentParser()
parser.add_argument("agent", help="the name of agent to simulate")
args = parser.parse_args()


def main():
    """
        Environment settings
            - single user
            - single service selected
            - one-to-one matching between services and devices
            - partial observation based on Euclidean-distance
            - 3-dimensional space
    """
    width = 200
    height = 10
    depth = 3
    max_speed = 1
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    conf = Configuration(num_user=1,
                         num_service=100,
                         width=width,
                         height=height,
                         depth=depth,
                         max_speed=max_speed,
                         observation=EuclideanObservation(observation_range=10),
                         #observation=FullObservation(),

                         user_constructor=UserConstructor(width, height, depth, max_speed),
                         service_constructor=VisualServiceConstructor(width, height, depth),
                         reward_function=HandoverPenaltyRewardFunction(effectiveness=VisualEffectiveness(
                             text_size_pixel=27,
                             resolution=1080,
                             visual_angle_min=5/60,
                             FoV_angle_max=105,
                             face_angle_max=90
                         )),

                         num_episode=1000,
                         num_step=100,
                         memory_size=1000,
                         batch_size=100,
                         learning_rate=1e-9,
                         discount_factor=.95,
                         eps_init=1.0,
                         eps_final=1e-2,
                         # set decaying rate according to the number of episodes: to make epsilon reaches eps_final at the end
                         eps_decay=np.power(1e-2/1.0, 1 / 1000),
                         agent=args.agent,
                         datetime=now,
                         summary_path=get_summary_path(agent=args.agent, date=now, filename="configuration.txt"))

    """ unit of distance is Meter """
    experiment = EffectDrivenMediumSelectionExperiment(configuration=conf)
    experiment.run()


if __name__ == '__main__':
    main()
