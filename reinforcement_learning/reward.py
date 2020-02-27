import tensorflow as tf
import json
from abc import abstractmethod

from models.effectiveness import Effectiveness
from utils import variable_summaries


class Reward:
    """ Reward: single reward signal """
    @abstractmethod
    def get_overall_score(self):
        pass

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


class PenaltyReward(Reward):
    """ PenaltyReward: single reward instance for combining penalty and effectiveness """
    def __init__(self, penalty, effectiveness):
        self.penalty = penalty
        self.effectiveness = effectiveness

    def get_overall_score(self):
        return float(self.penalty + self.effectiveness)


class RewardFunction:
    """ RewardFunction: abstract class for reward signal models """
    @abstractmethod
    def measure(self, user, service, context=None):
        pass

    @abstractmethod
    def __str__(self):
        pass


class HandoverPenaltyRewardFunction(RewardFunction):
    """ HandoverPenaltyRewardFunction: giving penalty when handover, otherwise effectiveness """
    def __init__(self, effectiveness):
        assert isinstance(effectiveness, Effectiveness)
        self.effectiveness = effectiveness

    def __str__(self):
        return "HandoverPenaltyReward({effectiveness}: {factors}".format(
            effectiveness=type(self.effectiveness).__name__,
            factors=json.dumps(self.effectiveness.__dict__)
        )

    def measure(self, user, service, context=None):
        """ Handover """
        if not (service.in_use and service.user == user) and user.service:
            penalty = -1
        else:
            penalty = 0

        return PenaltyReward(penalty=penalty, effectiveness=self.effectiveness.measure(user, service, context))

