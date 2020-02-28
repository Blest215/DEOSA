from models.entity.entity import Entity
from models.entity.device import DisplayDevice
from models.physics.coordinate import generate_random_coordinate
from models.physics.mobility import StaticMobility
from models.physics.orientation import generate_random_orientation


class Service(Entity):
    """ Service: a basic class that represents service instances """
    def __init__(self, name, service_type, device=None):
        self.name = name
        self.type = service_type

        # TODO currently, service - device is one-to-one matching
        if device:
            assert isinstance(device, DisplayDevice)
            self.device = device

        """ Flag: whether the service is in use or not """
        self.in_use = False
        self.user = None

    def __str__(self):
        return "Service name {name}, type {type}, device {device}".format(name=self.name,
                                                                          type=self.type,
                                                                          device=self.device)

    def acquire(self, user):
        """ acquire: user acquires the service to use """
        self.in_use = True
        self.user = user

    def release(self):
        """ release: user releases the service """
        self.in_use = False
        self.user = None

    def update(self):
        """ update: updates the states of the service, including device status """
        self.device.update()

    def vectorize(self):
        """ vectorize: returns vector representation of the service """
        # TODO vector representation of services: multi-user situation
        if self.in_use:
            return self.device.vectorize() + [1]
        else:
            return self.device.vectorize() + [0]


class VisualServiceConstructor:
    """
    VisualServiceConstructor: constructor class of visual services that utilizes DisplayDevice

    coordinate: random, restricted in rectangular space
    mobility: static
    orientation: random
    """
    def __init__(self, width, height, depth):
        self.width = width
        self.height = height
        self.depth = depth

        self.device_size = 1

    def get(self, index):
        # TODO currently, service is a simple encapsulation of device functionality, so device_type == service_type
        coordinate = generate_random_coordinate(self.width, self.height, self.depth)
        mobility = StaticMobility()
        orientation = generate_random_orientation()
        new_device = DisplayDevice(name=index,
                                   coordinate=coordinate,
                                   mobility=mobility,
                                   orientation=orientation,
                                   size=self.device_size)
        new_service = Service(name=index,
                              service_type="visual",
                              device=new_device)
        return new_service

    def __str__(self):
        return "VisualServiceConstructor(static, size={size})".format(size=self.device_size)