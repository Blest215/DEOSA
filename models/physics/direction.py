import random

import numpy as np

from models.math import Vector


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