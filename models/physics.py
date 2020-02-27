import random
import numpy as np
from abc import abstractmethod

from models.math import Vector, Quaternion
from utils import clamp


class Coordinate(Vector):
    """ Coordinate: class that represents coordinate of a physical entity in a 3-dimensional space """
    def get_distance(self, other):
        """ get_distance: calculates Euclidean distance between coordinates"""
        assert isinstance(other, Coordinate)
        return np.sqrt(np.square(self.x - other.x) + np.square(self.y - other.y) + np.square(self.z - other.z))

    def vectorize(self, width=200, height=10, depth=3):
        """ vectorize: returns a list form of the coordinate """
        # TODO normalization
        return [self.x/width, self.y/height, self.z/depth]

    def get_relative_coordinate(self, other):
        """ get_relative_coordinate: returns the relative coordinate of another coordinate, centered by itself """
        assert isinstance(other, Coordinate)
        return other - self


def generate_random_coordinate(width, height, depth):
    return Coordinate(x=random.random() * width, y=random.random() * height, z=random.random() * depth)


def generate_center_coordinate(width, height, depth):
    return Coordinate(x=width/2, y=height/2, z=depth/2)


def generate_custom_coordinate(width, height, depth, x, y, z):
    assert x <= width and y <= height and z <= depth
    return Coordinate(x=x, y=y, z=z)


class Direction(Vector):
    """ Direction: class that represents direction of a physical entity in a 3-dimensional space """
    def __init__(self, x, y, z):
        """ Direction should be a unit vector or zero vector """
        if x == 0 and y == 0 and z == 0:
            Vector.__init__(self, x=x, y=y, z=z)
        else:
            denominator = np.sqrt(np.square(x) + np.square(y) + np.square(z))
            Vector.__init__(self, x=x / denominator, y=y / denominator, z=z / denominator)
        assert self.size() == 1 or self.size() == 0


def generate_random_direction():
    return Direction(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1))


def generate_horizontal_direction():
    return Direction(random.uniform(-1, 1), random.uniform(-1, 1), 0)


def generate_custom_direction(x, y, z):
    return Direction(x=x, y=y, z=z)


class Mobility:
    """ Mobility: class that represents mobility of a physical entity """
    def __init__(self, direction, speed):
        """ direction: direction of the mobility, Direction class """
        assert isinstance(direction, Direction)
        self.direction = direction
        self.speed = speed

    @abstractmethod
    def update(self, coordinate):
        """ update: receives current coordinate and returns new """
        return coordinate

    def vectorize(self):
        """ vectorize: returns list form of the mobility, for concatenation with other lists """
        # return (self.speed * self.direction).vectorize() TODO assume static speed
        return self.direction.vectorize()


class RectangularDirectedMobility(Mobility):
    """ RectangularDirectedMobility: mobility that has fixed direction and speed, restricted in a rectangular area"""
    def __init__(self, width, height, depth, direction, speed):
        Mobility.__init__(self, direction, speed)
        self.width = width
        self.height = height
        self.depth = depth

    def update(self, coordinate):
        new_x = clamp(coordinate.x + self.direction.x * self.speed, 0, self.width)
        new_y = clamp(coordinate.y + self.direction.y * self.speed, 0, self.height)
        new_z = clamp(coordinate.z + self.direction.z * self.speed, 0, self.depth)

        if new_x == 0 or new_x == self.width or new_y == 0 or new_y == self.height or new_z == 0 or new_z == self.depth:
            """ if new direction is on the boundary of the area, reset direction randomly """
            self.direction = generate_random_direction()

        coordinate.update(new_x, new_y, new_z)

    def __str__(self):
        return "DirectedMobility(direction: {direction}, speed: {speed})".format(direction=self.direction,
                                                                                 speed=self.speed)


def generate_random_direction_random_speed_mobility(width, height, depth, max_speed):
    return RectangularDirectedMobility(width, height, depth,
                                       generate_random_direction(),
                                       random.random() * max_speed)


def generate_random_direction_specific_speed_mobility(width, height, depth, speed):
    return RectangularDirectedMobility(width, height, depth,
                                       generate_random_direction(),
                                       speed)


def generate_horizontal_direction_specific_speed_mobility(width, height, depth, speed):
    return RectangularDirectedMobility(width, height, depth,
                                       generate_horizontal_direction(),
                                       speed)


def generate_custom_mobility(width, height, depth, direction, speed):
    return RectangularDirectedMobility(width=width, height=height, depth=depth, direction=direction, speed=speed)


class StaticMobility(Mobility):
    def __init__(self):
        Mobility.__init__(self, Direction(0., 0., 0.), 0.)

    def update(self, coordinate):
        return coordinate

    def __str__(self):
        return "StaticMobility"


class Rotation(Quaternion):
    """ Rotation: class of rotation quaternion, receives axis and angle, then construct unit rotation vector """
    def __init__(self, theta, i, j, k):
        assert -2 * np.pi <= theta <= 2 * np.pi  # theta is radian

        """ Rotation should be a unit vector """
        denominator = np.square(i) + np.square(j) + np.square(k)
        Quaternion.__init__(
            self,
            w=np.cos(theta / 2),
            i=np.sign(i) * np.sqrt(np.square(i) / denominator) * np.sin(theta / 2),
            j=np.sign(j) * np.sqrt(np.square(j) / denominator) * np.sin(theta / 2),
            k=np.sign(k) * np.sqrt(np.square(k) / denominator) * np.sin(theta / 2)
        )

    def rotate(self, quaternion):
        return self * quaternion * self.get_conjugate()


class Orientation:
    """
        Orientation: class that represents orientation of a physical entity in a 3-dimensional space
        to un-ambiguously state orientation and head of a body,
        orientation receives a Quaternion and rotate vectors (1, 0, 0) and (0, 0, 1) according to the Quaternion,
        where each vector is face and head, respectively
    """
    def __init__(self, theta, i, j, k):
        rotation = Rotation(theta, i, j, k)

        default_face = Quaternion(0, 1, 0, 0)  # x-axis direction
        default_head = Quaternion(0, 0, 0, 1)  # z-axis direction

        self.face = rotation.rotate(default_face)
        self.head = rotation.rotate(default_head)

    def __str__(self):
        return "Orientation face:{face} | head:{head}".format(face=self.face, head=self.head)

    def vectorize(self):
        return self.face.get_vector_part().vectorize()  # TODO skip head  + self.head.get_vector_part().vectorize()


def generate_random_orientation():
    return Orientation(random.uniform(-2, 2) * np.pi, random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1))


def generate_random_vertical_orientation():
    """ Rotating axis is z-axis, so Orientation head always (0, 0, 1) """
    return Orientation(random.uniform(-2, 2) * np.pi, 0, 0, 1)


def generate_random_half_line_orientation(width, height, depth, x, y, z):
    """ orientation that faces half-line of the environment """
    orientation = generate_random_vertical_orientation()
    while (height/2 - y) * orientation.face.get_vector_part().y < 0:
        orientation = generate_random_vertical_orientation()
    return orientation
