import numpy as np

from models.math import Vector


class Direction(Vector):
    """ Direction: class that represents direction of a physical entity in a 3-dimensional space """
    def __init__(self, x, y, z):
        assert (x is None or -1 <= x <= 1) and (y is None or -1 <= y <= 1) and (z is None or -1 <= z <= 1)
        """ Direction should be a unit vector or zero vector """
        if x == 0 and y == 0 and z == 0:
            Vector.__init__(self, x=x, y=y, z=z)
        else:
            # Randomize if the value is None
            if x is None:
                x = np.random.uniform(-1, 1)
            if y is None:
                y = np.random.uniform(-1, 1)
            if z is None:
                z = np.random.uniform(-1, 1)
            denominator = np.sqrt(np.square(x) + np.square(y) + np.square(z))
            Vector.__init__(self, x=x / denominator, y=y / denominator, z=z / denominator)
        assert np.isclose(self.size(), 1) or np.isclose(self.size(), 0)

    def rotate_xy(self, radian):
        """ rotate_xy: rotate the vector along the xy-plane, ignores z """
        cos = np.cos(radian)
        sin = np.sin(radian)
        return Direction(x=self.x * cos - self.y * sin, y=self.x * sin + self.y * cos, z=self.z)
