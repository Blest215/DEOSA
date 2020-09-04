import argparse
import datetime
import os

from experiment import Experiment
from models.environment.environment import Environment
from reinforcement_learning.reward import HandoverPenaltyRewardFunction
from reinforcement_learning.agent.baselines import RandomSelectionAgent, NearestSelectionAgent, \
    NoHandoverSelectionAgent, GreedySelectionAgent
from reinforcement_learning.agent.dqn import EDMSAgentDQN

from models.environment.observation import EuclideanObservation
from models.effectiveness import VisualEffectiveness
from models.entity.service import VisualServiceConstructor
from models.entity.user import UserConstructor

parser = argparse.ArgumentParser()
parser.add_argument("agent", help="the name of agent to simulate")
args = parser.parse_args()


# tf.keras.backend.set_floatx('float64')


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
    height = 20
    depth = 3
    max_speed = 1
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    env = Environment(num_user=1,
                      num_service=100,
                      width=width,
                      height=height,
                      depth=depth,
                      observation=EuclideanObservation(observation_range=10),
                      user_constructor=UserConstructor(width, height, depth, max_speed),
                      service_constructor=VisualServiceConstructor(width, height, depth),
                      reward_function=HandoverPenaltyRewardFunction(effectiveness=VisualEffectiveness(
                          text_size_pixel=27,
                          resolution=1080,
                          visual_angle_min=5 / 60,
                          FoV_angle_max=105,
                          face_angle_max=90
                      )))

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
    if args.agent == "EDMS(DQN)":
        agent = EDMSAgentDQN(env, now,
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
                            num_episode=10000,
                            num_step=10,
                            now=now)

    experiment.reset()
    experiment.run()


if __name__ == '__main__':
    main()
