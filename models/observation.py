from abc import abstractmethod


class Observation:
    """ Observation: abstract class for Observation models """

    env = None

    def set_environment(self, env):
        """ set_environment: set target environment to get observation """
        self.env = env

    @abstractmethod
    def get_observation(self):
        """ get_observation: return the observation according to the state of the environment """
        pass

    @abstractmethod
    def get_observation_vector(self):
        """ get_observation_vector: return the observation in vector format """
        pass

    @abstractmethod
    def __str__(self):
        pass


class EuclideanObservation(Observation):
    """ EuclideanObservation: class that calculates circular partial observation based on Euclidean distance """
    def __init__(self, observation_range):
        self.observation_range = observation_range

    def get_observation(self):
        """ get_observation: returns the Euclidean-distance-based partial observation on the environment """
        service_observation = [service for service in self.env.services
                               if self.env.user.get_distance(service.device) <= self.observation_range]
        """ return objects, rather than matrix: agent will transform the observation into matrix """
        return {
            "user": self.env.user,
            "services": service_observation
        }

    def get_observation_vector(self):
        """ get_observation_vector: returns the Euclidean-distance-based partial observation on the environment """
        observation = self.get_observation()
        return {
            "user": observation["user"].vectorize(),
            "services": [service.vectorize() for service in observation["services"]]
        }

    def __str__(self):
        return "EuclideanObservation({range})".format(range=self.observation_range)


class FullObservation(Observation):
    def get_observation(self):
        """ get_observation: returns the Full observation """
        return {
            "user": self.env.user,
            "services": self.env.services
        }

    def get_observation_vector(self):
        """ get_observation_vector: returns the Full observation """
        observation = self.get_observation()
        return {
            "user": observation["user"].vectorize(),
            "services": [service.vectorize() for service in observation["services"]]
        }

    def __str__(self):
        return "FullObservation"
