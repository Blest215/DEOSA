from models.physics.mobility import Mobility
from models.physics.coordinate import Coordinate


class Entity:
    """ Entity: a very basic class for any cyber or physical entity """

    def vectorize(self):
        """ vectorize: returns vector representation of the entity """
        return []

    def update(self):
        """ update: updates the states of the entity, call for each environment step """
        pass


class PhysicalEntity(Entity):
    """ PhysicalEntity: physical body class, mainly deals with coordinate and mobility """
    def __init__(self, coordinate, mobility):
        assert isinstance(mobility, Mobility)
        self.mobility = mobility

        assert isinstance(coordinate, Coordinate)
        self.coordinate = coordinate

    def get_coordinate(self):
        """ get_coordinate: get current coordinate of the body """
        return self.coordinate.unpack()

    def get_distance(self, other):
        """ distance: calculate distance from another Body"""
        assert isinstance(other, PhysicalEntity)
        return self.coordinate.get_distance(other.coordinate)

    def move(self):
        """ move: update the coordinate of the body according to its mobility """
        self.mobility.update(self.coordinate)


