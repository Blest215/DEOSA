import random

import numpy as np

from models.math import Vector


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
    return Coordinate(x=width / 2, y=height / 2, z=depth / 2)


def generate_custom_coordinate(width, height, depth, x, y, z):
    assert x <= width and y <= height and z <= depth
    return Coordinate(x=x, y=y, z=z)
