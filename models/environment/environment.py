import matplotlib.pyplot as plt

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
                     coordinate=Coordinate(x=10, y=self.height / 2, z=1.7),
                     # Orientation is same with mobility direction
                     orientation=direction,
                     # Go across the environment
                     mobility=RectangularDirectedMobility(self.width, self.height, self.depth,
                                                          direction, self.max_speed),
                     visual_acuity=0.0)
            )
        """ the first user becomes primary (main) user """
        self.user = self.users[0]

        """ set devices and services in the environment """
        for i in range(self.num_service):
            self.services.append(
                VisualOutputService(
                    sid=i,
                    location=generate_random_coordinate(self.width, self.height, self.depth),
                    orientation=Direction(None, None, 0),
                    text_size=24  # TODO constant
                )
            )

        # if not self.get_observation()["services"]:
        #     """ reset until at least one service discovered """
        #     return self.reset()

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

    def get_observation_size(self):
        return len(self.user.vectorize())

    def get_action_size(self):
        return len(self.services[0].vectorize())

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

        if not self.get_observation()["services"]:
            """ if no service is discovered, done is True """
            done = True

        return self.get_observation(), reward, done

    def render(self):  # TODO
        fig = plt.figure()

        # locations of user and devices
        plt.scatter(x=[device.coordinate.x for device in self.devices] + [self.user.coordinate.x],
                    y=[device.coordinate.y for device in self.devices] + [self.user.coordinate.y],
                    c=["blue" for _ in range(self.num_service)] + ["red"])

        # orientations of user and devices
        head_width = 0.05
        head_length = 0.05
        for device in self.devices:
            plt.arrow(x=device.coordinate.x, y=device.coordinate.y,
                      dx=device.orientation.face.i, dy=device.orientation.face.j,
                      head_width=head_width, head_length=head_length)
        plt.arrow(x=self.user.coordinate.x, y=self.user.coordinate.y,
                  dx=self.user.infer_orientation().x, dy=self.user.infer_orientation().y,
                  head_width=head_width, head_length=head_length)

        # TODO observation range

        plt.show()
