import numpy as np

from reinforcement_learning.agent.agent import Agent


class RandomSelectionAgent(Agent):
    """ RandomSelectionAgent: a baseline agent that selects services randomly """
    def selection(self, user, services):
        index = np.random.choice(range(len(services)))
        return services[index], index


class NearestSelectionAgent(Agent):
    """ ClosestSelectionAgent: a baseline agent that selects the nearest service"""
    def selection(self, user, services):
        minimum = 1000000
        index = -1
        for i in range(len(services)):
            if user.get_distance(services[i].device) < minimum:
                index = i
                minimum = user.get_distance(services[i].device)
        return services[index], index


class NoHandoverSelectionAgent(Agent):
    """ NoHandoverSelectionAgent: a baseline agent that minimizes the number of handovers """
    def selection(self, user, services):
        for i in range(len(services)):
            if services[i].in_use and services[i].user == user:
                return services[i], i
        """ if no service is currently in-use, select randomly """
        index = np.random.choice(range(len(services)))
        return services[index], index


class GreedySelectionAgent(Agent):
    """ GreedySelectionAgent: a baseline agent that selects best one currently, in terms of effectiveness """
    def selection(self, user, services):
        def measure(service):
            relative_location = service.device.location - user.location

            """ 
                Orientation 
                face of the visual display should be opposite of the user's face
            """
            device_orientation = service.device.orientation
            psi = device_orientation.get_angle(-relative_location)
            if psi > 70:
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
            if visual_angle / 5 < user.minimum_visual_angle:
                return 0

            return 1

        effectiveness = [measure(service) for service in services]
        maximum = np.max(effectiveness)
        effective_services_index = np.where(effectiveness == maximum)[0]

        for i in effective_services_index:
            if services[i].in_use and services[i].user == user:
                return services[i], i
        index = np.random.choice(effective_services_index)
        return services[index], index
