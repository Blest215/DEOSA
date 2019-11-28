import numpy as np
import random
from abc import abstractmethod

from models.mobility import Coordinate, Mobility, StaticMobility
from models.orientation import Orientation, Rotation
from models.math import Vector


class Service:
    """ Service: a basic class that represents service instances """
    def __init__(self, name, service_type, device=None):
        self.name = name
        self.type = service_type

        """ Type of associated device should be Device """
        # TODO currently, service - device is one-to-one matching
        assert isinstance(device, Device)
        self.device = device

        """ Flag: whether the service is in use or not """
        self.in_use = False
        self.user = None

    def __str__(self):
        return "Service name {name}, type {type}, device {device}".format(name=self.name,
                                                                          type=self.type,
                                                                          device=self.device)

    def acquire(self, user):
        """ acquire: user acquires the service to use """
        assert isinstance(user, User)
        self.in_use = True
        self.user = user

    def release(self):
        """ release: user releases the service """
        self.in_use = False
        self.user = None

    def update(self):
        self.device.update()

    def vectorize(self):
        # TODO vector representation of services: multi-user situation
        if self.in_use:
            return self.device.vectorize() + [1]
        else:
            return self.device.vectorize() + [0]


class Body:
    """ Body: physical body class, mainly deals with coordinate and mobility """
    def __init__(self, coordinate, mobility):
        """ mobility of the body """
        assert isinstance(mobility, Mobility)
        self.mobility = mobility

        """ coordinate of the body """
        assert isinstance(coordinate, Coordinate)
        self.coordinate = coordinate

    def get_coordinate(self):
        return self.coordinate.get()

    def distance(self, other):
        assert isinstance(other, Body)
        return self.coordinate.distance(other.coordinate)

    def move(self):
        self.mobility.update(self.coordinate)

    @abstractmethod
    def vectorize(self):
        return []


class Device(Body):
    """ Device: a basic class that represents devices """

    def __init__(self, name, device_type, coordinate, mobility, orientation, size):
        Body.__init__(self, coordinate, mobility)
        self.name = name
        self.type = device_type

        assert isinstance(orientation, Orientation)
        self.orientation = orientation

        self.size = size

    def __str__(self):
        return "Device {name}, type {type} at {coordinate}, {orientation}".format(name=self.name,
                                                                                  type=self.type,
                                                                                  coordinate=self.coordinate,
                                                                                  orientation=self.orientation)

    def update(self):
        self.move()

    def vectorize(self):
        if isinstance(self.mobility, StaticMobility):
            # If mobility is static, skip to put mobility information
            return self.coordinate.vectorize() + self.orientation.vectorize() + [self.size]
        return self.coordinate.vectorize() + self.orientation.vectorize() + self.mobility.vectorize() + [self.size]


class User(Body):
    """ User: a basic class that represents users """
    def __init__(self, uid, coordinate, mobility):
        Body.__init__(self, coordinate, mobility)
        self.uid = uid
        self.service = None

        self.orientation = None
        self.update_orientation()

    def __str__(self):
        return "User {uid} at {coordinate} orientation {face}".format(uid=self.uid,
                                                                      coordinate=self.coordinate,
                                                                      face=self.infer_orientation())

    def utilize(self, service):
        assert isinstance(service, Service)
        self.service = service

    def infer_orientation(self):
        return self.orientation.get_vector_part()

    def update_orientation(self):
        """ update orientation of user head from mobility """
        # Orientation of the user is randomly generated based on the mobility
        # TODO orientation here is different from Orientation
        mobility_orientation = self.mobility.direction.to_quaternion()
        random_horizontal_rotation = Rotation(np.random.normal(loc=0.0, scale=0.2), 0, 0, 1)
        vertical_rotation_axis = self.mobility.direction.cross(Vector(0, 0, 1))
        random_vertical_rotation = Rotation(np.random.normal(loc=0.0, scale=0.2),
                                            vertical_rotation_axis.x,
                                            vertical_rotation_axis.y,
                                            vertical_rotation_axis.z)
        user_orientation = random_horizontal_rotation.rotate(
            random_vertical_rotation.rotate(mobility_orientation)
        )
        self.orientation = user_orientation

    def update(self):
        self.move()
        self.update_orientation()

    def vectorize(self):
        return self.coordinate.vectorize() + self.mobility.vectorize()
