from abc import abstractmethod


class Observation:
    """ Observation: abstract class for Observation models """
    # TODO revision required

    @abstractmethod
    def get_observation(self, env):
        """ get_observation: return the observation according to the state of the environment """
        return {
            "user": None,
            "services": []
        }

    def get_observation_vector(self, env):
        """ get_observation_vector: return the observation in vector format """
        observation = self.get_observation(env)
        return {
            "user": observation["user"].vectorize(),
            "services": [service.vectorize() for service in observation["services"]]
        }

    @abstractmethod
    def __str__(self):
        return ""


class EuclideanObservation(Observation):
    """ EuclideanObservation: class that calculates circular partial observation based on Euclidean distance """
    def __init__(self, observation_range):
        self.observation_range = observation_range

    def get_observation(self, env):
        """ get_observation: returns the Euclidean-distance-based partial observation on the environment """
        service_observation = [service for service in env.services
                               if env.user.get_distance(service.device) <= self.observation_range]
        """ return objects, rather than matrix: agent will transform the observation into matrix """
        return {
            "user": env.user,
            "services": service_observation
        }

    def __str__(self):
        return "EuclideanObservation({range})".format(range=self.observation_range)


class FullObservation(Observation):
    def get_observation(self, env):
        """ get_observation: returns the Full observation """
        return {
            "user": env.user,
            "services": env.services
        }

    def __str__(self):
        return "FullObservation"
