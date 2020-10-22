import numpy as np

from models.entity.entity import PhysicalEntity
from models.math import Rotation, Vector


class User(PhysicalEntity):
    """ User: a basic class that represents users """

    def __init__(self, uid, coordinate, orientation, mobility, visual_acuity):
        PhysicalEntity.__init__(self, coordinate, orientation, mobility)
        self.id = uid
        self.visual_acuity = visual_acuity
        self.service = None

    def __str__(self):
        return "User {uid} at {coordinate} orientation {face} ({acuity})".format(uid=self.id,
                                                                                 coordinate=self.location,
                                                                                 face=self.orientation(),
                                                                                 acuity=self.visual_acuity)

    def utilize(self, service):
        if self.service:
            self.service.release()
        self.service = service
        service.acquire(self)

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
