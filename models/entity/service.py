from abc import abstractmethod

from models.entity.entity import Entity, PhysicalEntity
from models.physics.mobility import StaticMobility


class Service(Entity):
    """ Service: a basic class that represents service instances """

    def __init__(self, sid):
        self.id = sid

        """ Flag: whether the service is in use or not """
        self.in_use = False
        self.user = None

    def __str__(self):
        return "{type} id {id}".format(type=type(self), id=self.id)

    def acquire(self, user):
        """ acquire: user acquires the service to use """
        self.in_use = True
        self.user = user

    def release(self):
        """ release: user releases the service """
        self.in_use = False
        self.user = None

    def update(self):
        return None

    def vectorize(self):
        return [1 if self.in_use else 0]


class VisualOutputService(Service):

    def __init__(self, sid, location, orientation, text_size, scaling_constant):
        super().__init__(sid)

        # Device-related
        # TODO currently, service - device is one-to-one matching
        self.device = PhysicalEntity(location, orientation, StaticMobility())

        # Content-related
        self.text_size = text_size
        self.scaling_constant = scaling_constant

    def update(self):
        self.device.update()

    def vectorize(self):
        # TODO vector representation of services: multi-user situation
        return self.device.vectorize() + [self.scaling_constant] + super().vectorize()

    def __str__(self):
        return "Visual Output Service ({id}) at {location}, {orientation}. Text Size ({size})".format(id=self.id,
                                                                                                      location=self.device.location,
                                                                                                      orientation=self.device.orientation,
                                                                                                      size=self.text_size)
