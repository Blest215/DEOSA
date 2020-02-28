from models.entity.entity import Body
from models.physics.mobility import StaticMobility
from models.physics.orientation import Orientation


class DisplayDevice(Body):
    """ DisplayDevice: a class that represents display devices """

    def __init__(self, name, coordinate, mobility, orientation, size):
        Body.__init__(self, coordinate, mobility)
        self.name = name

        assert isinstance(orientation, Orientation)
        self.orientation = orientation

        self.size = size

    def __str__(self):
        return "Display device {name}, at {coordinate}, {orientation}".format(name=self.name,
                                                                              coordinate=self.coordinate,
                                                                              orientation=self.orientation)

    def update(self):
        self.move()

    def vectorize(self):
        if isinstance(self.mobility, StaticMobility):
            # If mobility is static, skip to put mobility information
            return self.coordinate.vectorize() + self.orientation.vectorize() + [self.size]
        return self.coordinate.vectorize() + self.orientation.vectorize() + self.mobility.vectorize() + [self.size]