import numpy as np

from models.math import Vector


class Coordinate(Vector):
    """ Coordinate: class that represents coordinate of a physical entity in a 3-dimensional space """

    def vectorize(self, width=200, height=10, depth=3):
        """ vectorize: returns a list form of the coordinate """
        # TODO normalization
        return [self.x, self.y, self.z]
        # return [self.x/width, self.y/height, self.z/depth]


def generate_random_coordinate(width, height, depth):
    return Coordinate(x=np.random.random() * width, y=np.random.random() * height, z=np.random.random() * depth)


def generate_center_coordinate(width, height, depth):
    return Coordinate(x=width / 2, y=height / 2, z=depth / 2)


def generate_custom_coordinate(width, height, depth, x, y, z):
    assert x <= width and y <= height and z <= depth
    return Coordinate(x=x, y=y, z=z)
