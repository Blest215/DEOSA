import random
from abc import abstractmethod

from utils import clamp
from models.physics.direction import Direction


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
        return self.direction.vectorize() + [self.speed]


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