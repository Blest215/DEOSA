import tensorflow as tf
import json
from abc import abstractmethod

from models.effectiveness import EffectivenessFunction


class Reward:
    """ Reward: single reward signal """
    def __init__(self, effectiveness, penalty, weight=None):
        self.effectiveness = float(effectiveness)
        self.penalty = float(penalty)
        assert weight is None or 0 <= weight <= 1
        self.weight = weight

    def get_overall_score(self):
        if self.weight:
            return float(self.effectiveness * self.weight + self.penalty * (1-self.weight))
        return float(self.effectiveness - self.penalty)

    def __add__(self, other):
        return float(self) + float(other)

    def __radd__(self, other):
        return float(other) + float(self)

    def __floordiv__(self, other):
        return float(self) // other

    def __truediv__(self, other):
        return float(self) / other

    def __float__(self):
        return float(self.get_overall_score())

    def __str__(self):
        return str(self.get_overall_score())

    def __gt__(self, other):
        if isinstance(other, Reward):
            return self.get_overall_score() > other.get_overall_score()
        return self.get_overall_score() > other

    def __ge__(self, other):
        if isinstance(other, Reward):
            return self.get_overall_score() >= other.get_overall_score()
        return self.get_overall_score() >= other

    def __lt__(self, other):
        if isinstance(other, Reward):
            return self.get_overall_score() < other.get_overall_score()
        return self.get_overall_score() < other

    def __le__(self, other):
        if isinstance(other, Reward):
            return self.get_overall_score() <= other.get_overall_score()
        return self.get_overall_score() <= other


class RewardFunction:
    """ RewardFunction: giving penalty when handover, otherwise effectiveness """
    def __init__(self, effectiveness_function, penalty, weight):
        assert isinstance(effectiveness_function, EffectivenessFunction)
        self.effectiveness_function = effectiveness_function
        self.penalty = penalty
        self.weight = weight

        self.__setting__ = self.__dict__.copy()
        self.__setting__["name"] = type(self).__name__

    def measure(self, user, service, context=None):
        """ Handover """
        if not (service.in_use and service.user == user) and user.service:
            penalty = self.penalty
        else:
            penalty = 0

        return Reward(effectiveness=self.effectiveness_function.measure(user, service, context),
                      penalty=penalty,
                      weight=self.weight)

