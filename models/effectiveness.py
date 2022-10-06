import numpy as np
from abc import abstractmethod

from models.entity.service import VisualOutputService
from models.entity.user import User


class EffectivenessFunction:
    """ Effectiveness: abstract class for defining service effectiveness models """
    @abstractmethod
    def measure(self, user, service, context=None):
        pass


class VisualEffectivenessFunction(EffectivenessFunction):
    """ VisualEffectivenessFunction: effectiveness model for visual services """
    def __init__(self, visual_field_max, viewing_angle_max):
        self.visual_field_max = visual_field_max
        self.viewing_angle_max = viewing_angle_max
        self.text_scaling_constant = 5

        self.__setting__ = self.__dict__.copy()

    def measure(self, user, service, context=None):
        """ measure: measure the visual service effectiveness """
        assert isinstance(user, User) and isinstance(service, VisualOutputService)

        """ 
            Visual Field
            Device should be inside of user's visual field 
        """
        user_orientation = user.orientation
        relative_location = service.device.location - user.location
        theta = relative_location.get_angle(user_orientation)
        if theta > self.visual_field_max:
            return 0

        """ 
            Orientation 
            face of the visual display should be opposite of the user's face
        """
        device_orientation = service.device.orientation
        psi = device_orientation.get_angle(-relative_location)
        if psi > self.viewing_angle_max:
            # angle between user sight and device face is larger than 60 degree
            return 0

        """ 
            Visual angle  
            6/6 vision is defined as: at 6 m distance, human can recognize 5 arc-min letter.
            so size of the minimum letter is: 2 * 6 * tan(5 / 120) = 0.00873 m  
        """
        text_size = service.text_size * service.scaling_constant * 0.000352778
        perceived_size = text_size * device_orientation.get_cosine_angle(-relative_location)  # cos(psi)
        visual_angle = np.degrees(2 * np.arctan(perceived_size / (2 * user.get_distance(service.device))))
        """
            "the size of a letter on the Snellen chart of Landolt C chart is a visual angle of 5 arc minutes"
            https://en.wikipedia.org/wiki/Visual_acuity 
        """
        if visual_angle / self.text_scaling_constant < user.minimum_visual_angle:
            return 0

        return 1
