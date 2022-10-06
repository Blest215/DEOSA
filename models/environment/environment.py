import matplotlib.pyplot as plt
import numpy as np

from models.entity.service import Service, VisualOutputService
from models.entity.user import User
from models.environment.observation import Observation
from models.physics.coordinate import Coordinate, generate_random_coordinate
from models.physics.direction import Direction
from models.physics.mobility import RectangularDirectedMobility
from reinforcement_learning.reward import RewardFunction


class Environment:
    """ Environment: abstract class of IoT environments for required methods """

    def __init__(self, num_user, num_service, width, height, depth, max_speed, observation, reward_function):
        """ __init__: initialize the environment by setting configurations and resetting """

        """ num_user: number of users """
        assert num_user >= 1
        self.num_user = num_user
        """ num_service: number of services """
        assert num_service >= 1
        self.num_service = num_service
        """ width: x-axis size of the environment """
        assert width > 0
        self.width = width
        """ height: y-axis size of the environment """
        assert height > 0
        self.height = height
        """ depth: z-axis size of the environment """
        assert depth > 0
        self.depth = depth
        """ max_speed: maximum speed that the entities can have """
        assert max_speed > 0
        self.max_speed = max_speed
        """ observation: observation model """
        assert isinstance(observation, Observation)
        self.observation = observation

        """ reward: reward model """
        assert isinstance(reward_function, RewardFunction)
        self.reward_function = reward_function

        self.user_height = 1.7
        self.user_visual_acuity = 0.0
        self.service_text_size = 24
        self.service_scaling_constant_min = 1.0
        self.service_scaling_constant_max = 5.0

        self.__setting__ = self.__dict__.copy()

        """ user: main user that utilizes services """
        self.user = None
        """ users: the list of users """
        self.users = []
        """ services: the list of services """
        self.services = []

        self.reset()

    def reset(self):
        """ reset: resets the environment """
        self.user = None
        self.users = []
        self.services = []

        """ set users """
        for i in range(self.num_user):
            direction = Direction(1, 0, 0)
            self.users.append(
                User(uid=i,
                     # Start from edge of the environment
                     coordinate=Coordinate(x=0, y=self.height / 2, z=self.user_height),
                     # Orientation is same with mobility direction
                     orientation=direction,
                     # Go across the environment
                     mobility=RectangularDirectedMobility(self.width, self.height, self.depth,
                                                          direction, self.max_speed),
                     visual_acuity=self.user_visual_acuity)
            )
            self.users[-1].update_orientation()
        """ the first user becomes primary (main) user """
        self.user = self.users[0]

        """ set devices and services in the environment """
        for i in range(self.num_service):
            self.services.append(
                VisualOutputService(
                    sid=i,
                    location=generate_random_coordinate(self.width, self.height, self.depth),
                    orientation=Direction(None, None, 0),
                    text_size=self.service_text_size,
                    scaling_constant=np.random.randint(self.service_scaling_constant_min,
                                                       self.service_scaling_constant_max+1)
                )
            )

        if not self.get_observation()["services"]:
            """ reset until at least one service discovered """
            return self.reset()

        return self.get_observation()

    def get_state(self):
        """ get_state: return the full state of the environment """
        return {
            "user": self.user,
            "services": self.services
        }

    def get_observation(self):
        """ return observation in both dictionary format """
        return self.observation.get_observation(self)

    def step(self, action):
        """ step: make environment one step further by perform selection and update states """

        """ receives selection result as a service instance """
        assert isinstance(action, Service)

        done = False
        """ measure reward value according to the selection """
        reward = self.reward_function.measure(self.user, action)

        """ Release service and acquire new """
        self.user.utilize(action)

        for user in self.users:
            user.update()
        for service in self.services:
            service.update()

        new_observation = self.get_observation()
        if not new_observation["services"]:
            """ if no service is discovered, done is True """
            done = True

        return new_observation, reward, done
