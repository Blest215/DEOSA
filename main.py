import argparse
import datetime
import tensorflow as tf

from experiment import Experiment
from models.environment.environment import Environment
from reinforcement_learning.reward import RewardFunction
from reinforcement_learning.agent.baselines import RandomSelectionAgent, NearestSelectionAgent, \
    NoHandoverSelectionAgent, GreedySelectionAgent
from reinforcement_learning.agent.deosa import DEOSA
from reinforcement_learning.network import DQNetwork, DoubleDQNetwork, MunchausenDQNetwork

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
                      height=10,  # y
                      depth=5,  # z
                      max_speed=2,
                      observation=FullObservation(),
                      reward_function=RewardFunction(effectiveness_function=VisualEffectivenessFunction(
                          visual_field_max=80,
                          viewing_angle_max=70
                      ), penalty=0.5, weight=None))

    experiment = Experiment(num_episode=1000,
                            num_step=50,
                            now=now)

    if args.agent == "random" or args.agent == "all":
        experiment.run(agent=RandomSelectionAgent("Random", env, now), train=False)

    if args.agent == "greedy" or args.agent == "all":
        experiment.run(agent=GreedySelectionAgent("Greedy", env, now), train=False)

    if args.agent == "nearest" or args.agent == "all":
        experiment.run(agent=NearestSelectionAgent("Nearest", env, now), train=False)

    if args.agent == "nohandover" or args.agent == "all":
        experiment.run(agent=NoHandoverSelectionAgent("NoHandover", env, now), train=False)

    if args.agent == "DEOSA" or args.agent == "all":
        learning_rate = 1e-3
        discount_factor = 0.99
        tau = 0.001
        temperature = 0.1
        alpha = 0.9
        hidden_units = [128, 128, 128, 128, 128]
        activation = 'relu'

        memory_size = 1000
        batch_size = 10
        eps_init = 1.0
        eps_final = 0.1
        eps_decay = 0.99

        experiment.run(
            agent=DEOSA(
                env, now,
                memory_size=memory_size,
                batch_size=batch_size,
                network=MunchausenDQNetwork(
                    learning_rate=learning_rate,
                    discount_factor=discount_factor,
                    tau=tau,
                    temperature=temperature,
                    alpha=alpha,
                    hidden_units=hidden_units,
                    activation=activation
                ),
                eps_init=eps_init,
                eps_final=eps_final,
                eps_decay=eps_decay
            ),
            train=True
        )

        experiment.run(
            agent=DEOSA(
                env, now,
                memory_size=memory_size,
                batch_size=batch_size,
                network=DoubleDQNetwork(
                    learning_rate=learning_rate,
                    discount_factor=discount_factor,
                    tau=tau,
                    temperature=temperature,
                    alpha=alpha,
                    hidden_units=hidden_units,
                    activation=activation
                ),
                eps_init=eps_init,
                eps_final=eps_final,
                eps_decay=eps_decay
            ),
            train=True
        )

        experiment.run(
            agent=DEOSA(
                env, now,
                memory_size=memory_size,
                batch_size=batch_size,
                network=DQNetwork(
                    learning_rate=learning_rate,
                    discount_factor=discount_factor,
                    tau=tau,
                    temperature=temperature,
                    alpha=alpha,
                    hidden_units=hidden_units,
                    activation=activation
                ),
                eps_init=eps_init,
                eps_final=eps_final,
                eps_decay=eps_decay
            ),
            train=True
        )


if __name__ == '__main__':
    main()
