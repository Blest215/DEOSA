from abc import abstractmethod

from models.entity import User, Service, DisplayDevice
from models.physics import *
from models.physics import generate_random_orientation


class Constructor:
    """ Constructor: a basic class of constructor classes that constructs entities """

    @abstractmethod
    def get(self, index):
        pass

    @abstractmethod
    def __str__(self):
        pass


class UserConstructor(Constructor):
    """
    UserConstructor: constructor class of users

    coordinate: x=10 (near boundary of the space), y=height/2 (middle of the area), z=1.7 (common height of a human)
    """
    def __init__(self, width, height, depth, max_speed):
        self.width = width
        self.height = height
        self.depth = depth
        self.max_speed = max_speed

    def get(self, index):
        """ get: construct a User instance """
        return User(uid=index,
                    # Start from edge of the environment
                    coordinate=generate_custom_coordinate(self.width, self.height, self.depth,
                                                          x=10, y=self.height / 2,
                                                          z=1.7),
                    # Go across the environment
                    mobility=generate_custom_mobility(self.width,
                                                      self.height,
                                                      self.depth,
                                                      generate_custom_direction(1, 0, 0),
                                                      self.max_speed))

    def __str__(self):
        return "UserConstructor(x={x}, y={y}, z={z}, max_speed={max_speed})".format(x=10,
                                                                                    y="height/2",
                                                                                    z=1.7,
                                                                                    max_speed=self.max_speed)


class VisualServiceConstructor(Constructor):
    """
    VisualServiceConstructor: constructor class of visual services that utilizes DisplayDevice

    coordinate: random, restricted in rectangular space
    mobility: static
    orientation: random
    """
    def __init__(self, width, height, depth):
        self.width = width
        self.height = height
        self.depth = depth

        self.device_size = 1

    def get(self, index):
        # TODO currently, service is a simple encapsulation of device functionality, so device_type == service_type
        coordinate = generate_random_coordinate(self.width, self.height, self.depth)
        mobility = StaticMobility()
        orientation = generate_random_orientation()
        new_device = DisplayDevice(name=index,
                                   coordinate=coordinate,
                                   mobility=mobility,
                                   orientation=orientation,
                                   size=self.device_size)
        new_service = Service(name=index,
                              service_type="visual",
                              device=new_device)
        return new_service

    def __str__(self):
        return "VisualServiceConstructor(static, size={size})".format(size=self.device_size)
