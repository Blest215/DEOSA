from abc import abstractmethod

from models.physics.mobility import Mobility
from models.physics.coordinate import Coordinate
from models.physics.direction import Direction


class Entity:
    """ Entity: a very basic class for any cyber or physical entity """

    @abstractmethod
    def update(self):
        """ update: updates the states of the entity, call for each environment step """
        return None

    @abstractmethod
    def vectorize(self):
        """ vectorize: returns vector representation of the entity """
        return []


class PhysicalEntity(Entity):
    """ PhysicalEntity: physical body class, mainly deals with coordinate and mobility """
    def __init__(self, location, orientation, mobility):
        assert isinstance(location, Coordinate)
        self.location = location

        assert isinstance(orientation, Direction)
        self.orientation = orientation

        assert isinstance(mobility, Mobility)
        self.mobility = mobility

    def get_distance(self, other):
        """ distance: calculate distance from another Body"""
        assert isinstance(other, PhysicalEntity)
        return self.location.get_distance(other.location)

    def move(self):
        """ move: update the coordinate of the body according to its mobility """
        self.mobility.update(self.location)

    def update(self):
        self.move()

    def vectorize(self):
        return self.location.vectorize() + self.orientation.vectorize() + self.mobility.vectorize()
