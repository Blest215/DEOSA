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
        effectiveness = [self.env.reward_function.measure(user, service).effectiveness for service in services]
        maximum = np.max(effectiveness)
        effective_services_index = np.where(effectiveness == maximum)[0]

        for i in effective_services_index:
            if services[i].in_use and services[i].user == user:
                return services[i], i
        index = np.random.choice(effective_services_index)
        return services[index], index
