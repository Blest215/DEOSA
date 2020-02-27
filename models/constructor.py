from models.entity import User, Service, DisplayDevice
from models.physics import *
from models.physics import generate_random_orientation


def get_user_constructor(width, height, depth, max_speed):
    """
    get_user_constructor: returns constructor function of users

    coordinate: x=10 (near boundary of the space), y=height/2 (middle of the area), z=1.7 (common height of a human)
    """
    def user_constructor(index):
        """ user_constructor: construct a User instance """
        return User(uid=index,
                    # Start from edge of the environment
                    coordinate=generate_custom_coordinate(width, height, depth,
                                                          x=10, y=height / 2,
                                                          z=1.7),
                    # Go across the environment
                    mobility=generate_custom_mobility(width,
                                                      height,
                                                      depth,
                                                      generate_custom_direction(1, 0, 0),
                                                      max_speed))
    return user_constructor


def get_visual_service_constructor(width, height, depth):
    """
    get_visual_service_constructor: returns constructor function of visual services

    coordinate: random, restricted in rectangular space
    mobility: static
    orientation: random
    """
    def visual_service_constructor(index):
        """ service_constructor: constructs a Service instance that utilizes DisplayDevice """
        # TODO currently, service is a simple encapsulation of device functionality, so device_type == service_type
        coordinate = generate_random_coordinate(width, height, depth)
        mobility = StaticMobility()
        orientation = generate_random_orientation()
        new_device = DisplayDevice(name=index,
                                   coordinate=coordinate,
                                   mobility=mobility,
                                   orientation=orientation,
                                   size=1)
        new_service = Service(name=index,
                              service_type="visual",
                              device=new_device)
        return new_service
    return visual_service_constructor
