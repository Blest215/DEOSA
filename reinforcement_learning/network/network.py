import tensorflow as tf
from abc import abstractmethod


class Network(tf.keras.Model):
    def __init__(self, name, learning_rate, discount_factor, tau):
        super(Network, self).__init__(name=name)

        self.scope = "Network/{name}".format(name=name)
        self.learning_rate = learning_rate
        self.discount_factor = tf.constant(discount_factor, dtype=tf.float64)
        self.tau = tau
        self.target_network = None

    def set_target_network(self, target_network):
        """ set_target_network: sets target network to update from this network """
        assert type(self) == type(target_network) or target_network is None
        self.target_network = target_network

    def update_target_network(self):
        """ update_target_network: updates target network within given tau """
        assert self.target_network

        variables = self.trainable_variables
        target_variables = self.target_network.trainable_variables
        for v1, v2 in zip(variables, target_variables):
            v2.assign(self.tau * v1.numpy() + (1 - self.tau) * v2.numpy())

    def copy_from_target_network(self):
        """ copy_from_target_network: copies parameters from the target network """
        assert self.target_network

        variables = self.trainable_variables
        target_variables = self.target_network.trainable_variables
        for v1, v2 in zip(variables, target_variables):
            v1.assign(v2.numpy())

    def copy_to_target_network(self):
        """ copy_to_target_network: copies parameters to the target network """
        assert self.target_network

        variables = self.trainable_variables
        target_variables = self.target_network.trainable_variables
        for v1, v2 in zip(variables, target_variables):
            v2.assign(v1.numpy())

    def bootstrap(self, next_observation):
        """ bootstrap: bootstraps Q value of given next_observation and next_actions"""
        if self.target_network:
            """ if target network exist, bootstrapping from target network """
            return self.target_network(next_observation)
        return self(next_observation)

    @abstractmethod
    def call(self, observation, training=None, mask=None):
        """ calculate Q value of given observation and action """
        pass

    @abstractmethod
    def update(self, observation, action, reward, next_observation, done):
        """ update: updates network according to given observation, action, reward and next_observation value """
        pass

    def copy(self):
        # TODO
        pass


