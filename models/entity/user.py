import numpy as np

from models.entity.entity import PhysicalEntity
from models.math import Rotation, Vector
from models.physics.coordinate import generate_custom_coordinate
from models.physics.direction import generate_custom_direction
from models.physics.mobility import generate_custom_mobility


class User(PhysicalEntity):
    """ User: a basic class that represents users """
    def __init__(self, uid, coordinate, mobility):
        PhysicalEntity.__init__(self, coordinate, mobility)
        self.uid = uid
        self.service = None

        self.orientation = None
        self.update_orientation()

    def __str__(self):
        return "User {uid} at {coordinate} orientation {face}".format(uid=self.uid,
                                                                      coordinate=self.coordinate,
                                                                      face=self.infer_orientation())

    def utilize(self, service):
        self.service = service

    def infer_orientation(self):
        return self.orientation.get_vector_part()

    def update_orientation(self):
        """ update orientation of user head from mobility """
        # Orientation of the user is randomly generated based on the mobility
        # TODO orientation here is different from Orientation
        mobility_orientation = self.mobility.direction.to_quaternion()
        random_horizontal_rotation = Rotation(np.random.normal(loc=0.0, scale=0.2), 0, 0, 1)
        vertical_rotation_axis = self.mobility.direction.cross(Vector(0, 0, 1))
        random_vertical_rotation = Rotation(np.random.normal(loc=0.0, scale=0.2),
                                            vertical_rotation_axis.x,
                                            vertical_rotation_axis.y,
                                            vertical_rotation_axis.z)
        user_orientation = random_horizontal_rotation.rotate(
            random_vertical_rotation.rotate(mobility_orientation)
        )
        self.orientation = user_orientation

    def update(self):
        self.move()
        self.update_orientation()

    def vectorize(self):
        return self.coordinate.vectorize() + self.mobility.vectorize()


class UserConstructor:
    """
    UserConstructor: constructor class of users

    coordinate: x=10 (near boundary of the space), y=height/2 (middle of the area), z=1.7 (common height of a human)
    """
    def __init__(self, width, height, depth, max_speed):
        self.width = width
        self.height = height
        self.depth = depth
        self.max_speed = max_speed

    def get(self, index):
        """ get: construct a User instance """
        return User(uid=index,
                    # Start from edge of the environment
                    coordinate=generate_custom_coordinate(self.width, self.height, self.depth,
                                                          x=10, y=self.height / 2,
                                                          z=1.7),
                    # Go across the environment
                    mobility=generate_custom_mobility(self.width,
                                                      self.height,
                                                      self.depth,
                                                      generate_custom_direction(1, 0, 0),
                                                      self.max_speed))

    def __str__(self):
        return "UserConstructor(x={x}, y={y}, z={z}, max_speed={max_speed})".format(x=10,
                                                                                    y="height/2",
                                                                                    z=1.7,
                                                                                    max_speed=self.max_speed)