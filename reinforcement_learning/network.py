import tensorflow as tf
import numpy as np
from abc import abstractmethod

OBSERVATION_SIZE = 17


class Network(tf.keras.Model):
    def __init__(self, learning_rate, discount_factor, tau, temperature, alpha, hidden_units, activation):
        super(Network, self).__init__(name=type(self).__name__)
        self.scope = "Network/{name}".format(name=self.name)

        self.discount_factor = tf.constant(discount_factor, dtype=tf.float64)
        self.tau = tau
        self.temperature = tf.constant(temperature, dtype=tf.float64)
        self.alpha = tf.constant(alpha, dtype=tf.float64)
        self.hidden_units = hidden_units
        self.activation = activation

        self.learning_rate = learning_rate

        self.__setting__ = {
            'name': self.name,
            'learning_rate': self.learning_rate,
            'discount_factor': discount_factor,
            'tau': tau,
            'temperature': temperature,
            'alpha': alpha,
            'hidden_units': hidden_units,
            'activation': activation
        }

        self.target_network = None

        self.input_layer = tf.keras.layers.InputLayer(input_shape=(None, OBSERVATION_SIZE), name="InputLayer")

        with tf.name_scope("HiddenLayers"):
            self.hidden_layers = [
                tf.keras.layers.Dense(
                    unit, activation=activation
                ) for unit in hidden_units
            ]

        self.output_layer = tf.keras.layers.Dense(1, activation=None, name="OutputLayer")
        self.optimizer = tf.optimizers.Adam(self.learning_rate)
        self.build((None, None, OBSERVATION_SIZE))

    def create_target_network(self):
        """ create_target_network: creates target network automatically """
        self.target_network = self.copy()

    def set_target_network(self, target_network):
        """ set_target_network: sets target network to update from this network """
        assert type(self) == type(target_network) or target_network is None
        self.target_network = target_network

    def update_target_network(self):
        """ update_target_network: updates target network within given tau """
        if self.target_network:
            variables = self.trainable_variables
            target_variables = self.target_network.trainable_variables
            for v1, v2 in zip(variables, target_variables):
                v2.assign(self.tau * v1.numpy() + (1 - self.tau) * v2.numpy())

    def copy_from_target_network(self):
        """ copy_from_target_network: copies parameters from the target network """
        if self.target_network:
            variables = self.trainable_variables
            target_variables = self.target_network.trainable_variables
            for v1, v2 in zip(variables, target_variables):
                v1.assign(v2.numpy())

    def copy_to_target_network(self):
        """ copy_to_target_network: copies parameters to the target network """
        if self.target_network:
            variables = self.trainable_variables
            target_variables = self.target_network.trainable_variables
            for v1, v2 in zip(variables, target_variables):
                v2.assign(v1.numpy())

    def bootstrap(self, next_observation, target=True):
        """ bootstrap: bootstraps Q value of given next_observation and next_actions"""
        if target and self.target_network:
            """ if target network exist, bootstrapping from target network """
            return self.target_network(next_observation)
        return self(next_observation)

    @tf.function(input_signature=(  # input_signature is specified to avoid frequent retracing
            tf.TensorSpec(shape=[None, None, OBSERVATION_SIZE], dtype=tf.float64),
    ))
    def call(self, observation):
        z = self.input_layer(observation)
        for layer in self.hidden_layers:
            z = layer(z)
        output = self.output_layer(z)
        return output

    @tf.function(input_signature=(  # input_signature is specified to avoid frequent retracing
            tf.TensorSpec(shape=[None, None, OBSERVATION_SIZE], dtype=tf.float64),
            tf.TensorSpec(shape=[None], dtype=tf.int32),
            tf.TensorSpec(shape=[None], dtype=tf.float64),
            tf.TensorSpec(shape=[None, None, OBSERVATION_SIZE], dtype=tf.float64),
            tf.TensorSpec(shape=[None], dtype=tf.bool)
    ))
    def update(self, observations, actions, rewards, next_observations, done):
        """ update: updates network according to given observation, action, reward and next_observation value """
        batch_size = len(observations)
        target_Q = self.target(observations, actions, rewards, next_observations, done)

        with tf.GradientTape() as tape:
            indices = tf.stack([tf.range(batch_size), actions], axis=1)
            responsible_Q = tf.gather_nd(params=tf.squeeze(self.call(observations), axis=2),
                                         indices=indices,
                                         name="ResponsibleQ")

            loss = tf.reduce_sum(tf.square(tf.subtract(target_Q, responsible_Q)), name="Loss")

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

    def copy(self):
        return type(self)(self.learning_rate, self.discount_factor, self.tau, self.temperature, self.alpha,
                          self.hidden_units, self.activation)

    @abstractmethod
    def target(self, observations, actions, rewards, next_observations, done):
        """ target: calculate target Q value """
        pass

    @staticmethod
    def selection(Q, eps):
        """ epsilon-greedy """
        if np.random.random() < eps:
            selection = np.random.choice(range(len(Q)))
        else:
            selection = np.argmax(Q)

        return selection


class DQNetwork(Network):
    def target(self, observations, actions, rewards, next_observations, done):
        return tf.add(
            rewards,
            tf.squeeze(
                tf.multiply(
                    self.discount_factor,
                    tf.reduce_max(self.bootstrap(next_observations), axis=1)
                ), axis=1
            )
        )


class DoubleDQNetwork(Network):
    def target(self, observations, actions, rewards, next_observations, done):
        Q1 = tf.squeeze(self.bootstrap(next_observations, target=False), axis=2)
        Q2 = tf.squeeze(self.bootstrap(next_observations, target=True), axis=2)

        return tf.add(
            rewards,
            tf.multiply(
                self.discount_factor,
                tf.gather(params=Q2, indices=tf.argmax(Q1, axis=1), axis=1)
            )
        )


class MunchausenDQNetwork(Network):
    def target(self, observations, actions, rewards, next_observations, done):
        batch_size = len(observations)

        bootstrap = tf.squeeze(self.bootstrap(next_observations), axis=2)
        stochastic_policy = tf.nn.softmax(tf.divide(bootstrap, self.temperature), axis=1)

        indices = tf.stack([tf.range(batch_size), actions], axis=1)
        current_policy = tf.gather_nd(
            params=tf.nn.softmax(tf.divide(tf.squeeze(self.bootstrap(observations), axis=2), self.temperature), axis=1),
            indices=indices,
            name="CurrentQ")
        munchausen = tf.multiply(
            self.alpha,
            tf.multiply(self.temperature,
                        tf.clip_by_value(tf.math.log(current_policy), -1, 0))
        )

        maximum_entropy = tf.multiply(self.temperature, tf.clip_by_value(tf.math.log(stochastic_policy), -1, 0))

        return tf.add(
            tf.add(rewards, munchausen),
            tf.multiply(
                self.discount_factor,
                tf.reduce_sum(
                    tf.multiply(
                        stochastic_policy,
                        tf.subtract(
                            bootstrap,
                            maximum_entropy
                        )
                    ), axis=1)
            )
        )
