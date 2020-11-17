import numpy as np

from models.entity.entity import PhysicalEntity


class User(PhysicalEntity):
    """ User: a basic class that represents users """

    def __init__(self, uid, coordinate, orientation, mobility, visual_acuity):
        PhysicalEntity.__init__(self, coordinate, orientation, mobility)
        self.id = uid
        self.visual_acuity = visual_acuity
        self.minimum_visual_angle = pow(10, self.visual_acuity) / 60
        self.service = None

    def __str__(self):
        return "User {uid} at {coordinate} orientation {face} ({acuity})".format(uid=self.id,
                                                                                 coordinate=self.location,
                                                                                 face=self.orientation,
                                                                                 acuity=self.visual_acuity)

    def vectorize(self):
        return self.location.vectorize() + self.mobility.vectorize()

    def utilize(self, service):
        if self.service:
            self.service.release()
        self.service = service
        service.acquire(self)

    def update(self):
        self.move()
        self.update_orientation()

    def update_orientation(self):
        """ update orientation of user head from mobility """
        # restrict orientation about 30 degree (0.5 radian) from mobility direction
        rotation_angle_radian = np.random.normal(loc=0.0, scale=0.75)
        self.orientation = self.mobility.direction.rotate_xy(rotation_angle_radian)
