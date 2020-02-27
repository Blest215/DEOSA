import tensorflow as tf
import numpy as np
from abc import abstractmethod


class Network(tf.keras.Model):
    def __init__(self, name, learning_rate, discount_factor):
        super(Network, self).__init__()

        self.scope = "Network/{name}".format(name=name)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.target_network = None

    def set_target_network(self, target_network):
        """ set_target_network: sets target network to update from this network """
        assert type(self) == type(target_network)
        self.target_network = target_network

    def update_target_network(self, tau):
        """ update_target_network: updates target network within given tau """
        assert self.target_network and len(self.variables) == len(self.target_variables)

        variables = self.trainable_variables
        target_variables = self.target_network.trainable_variables
        for v1, v2 in zip(variables, target_variables):
            v2.assign(tau * v1.numpy() + (1 - tau) * v2.numpy())

    def copy_from_target_network(self):
        """ copy_from_target_network: copies parameters from the target network """
        assert self.target_network and len(self.variables) == len(self.target_variables)

        variables = self.trainable_variables
        target_variables = self.target_network.trainable_variables
        for v1, v2 in zip(variables, target_variables):
            v1.assign(v2.numpy())

    def bootstrap(self, next_observation):
        """ bootstrap: bootstraps Q value of given next_observation and next_actions"""
        if self.target_network:
            """ [DDQN] if target network exist, bootstrapping from target network """
            return self.target_network(next_observation)
        return self.call(next_observation)

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


class EDMSNetworkDQN(Network):
    """
     EDSSNetworkDQN network for variable size of action space

     Refer DRRN
     He, Ji, et al. "Deep reinforcement learning with an action space defined by natural language." (2016).
    """
    def __init__(self, observation_size, learning_rate, discount_factor, hidden_units, activation):
        super(EDMSNetworkDQN, self).__init__(name="EDMS(DQN)",
                                             learning_rate=learning_rate,
                                             discount_factor=discount_factor)

        self.input_layer = tf.keras.layers.InputLayer(input_shape=(observation_size,))
        self.hidden_layers = [
            tf.keras.layers.Dense(
                unit, activation=activation
            ) for unit in hidden_units
        ]
        self.output_layer = tf.keras.layers.Dense(1, activation=activation)

        self.optimizer = tf.optimizers.Adam(learning_rate)

    def call(self, observation, training=None, mask=None):
        z = self.input_layer(observation)
        for layer in self.hidden_layers:
            z = layer(z)
        output = self.output_layer(z)
        return output

    def update(self, observation, action, reward, next_observation, done):
        num_actions = len(observation)

        if done:
            """ if done, just add reward """
            target_Q = reward
        else:
            """ else, bootstrapping next Q value """
            target_Q = reward + self.discount_factor * np.max(self.bootstrap(next_observation))

        with tf.GradientTape() as tape:
            action_one_hot = tf.one_hot(action, num_actions, dtype=tf.float64)
            responsible_Q = tf.reduce_sum(tf.multiply(self.call(observation), action_one_hot))
            loss = tf.square(target_Q - responsible_Q)

            variables = self.trainable_variables
            gradients = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(zip(gradients, variables))

        return loss
