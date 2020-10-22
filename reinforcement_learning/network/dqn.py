import tensorflow as tf

from reinforcement_learning.network.network import Network


OBSERVATION_SIZE = 21


class DEOSANetwork(Network):
    """
     EDSSNetworkDQN network for variable size of action space

     Action (selection) is encoded as a vector and concatenated with user's state to form observation
     Q-value is calculated for each action and the service with the highest Q-value is selected

     Refer DRRN
     He, Ji, et al. "Deep reinforcement learning with an action space defined by natural language." (2016).
    """
    def __init__(self, learning_rate, discount_factor, hidden_units, activation):
        super(DEOSANetwork, self).__init__(name="DEOSANetwork",
                                           learning_rate=learning_rate,
                                           discount_factor=discount_factor)

        self.input_layer = tf.keras.layers.InputLayer(input_shape=(OBSERVATION_SIZE,))
        self.hidden_layers = [
            tf.keras.layers.Dense(
                unit, activation=activation
            ) for unit in hidden_units
        ]
        self.output_layer = tf.keras.layers.Dense(1, activation=activation)

        # self._set_inputs(inputs=self.input_layer)  # for saving model without compile or predict

        self.optimizer = tf.optimizers.Adam(learning_rate)

    @tf.function(input_signature=(  # input_signature is specified to avoid frequent retracing
            tf.TensorSpec(shape=[None, OBSERVATION_SIZE], dtype=tf.float64),
    ))
    def call(self, observation, training=None, mask=None):
        z = self.input_layer(observation)
        for layer in self.hidden_layers:
            z = layer(z)
        output = self.output_layer(z)
        return output

    @tf.function(input_signature=(  # input_signature is specified to avoid frequent retracing
            tf.TensorSpec(shape=[None, OBSERVATION_SIZE], dtype=tf.float64),
            tf.TensorSpec(shape=[], dtype=tf.int32),
            tf.TensorSpec(shape=[], dtype=tf.float64),
            tf.TensorSpec(shape=[None, OBSERVATION_SIZE], dtype=tf.float64),
            tf.TensorSpec(shape=[], dtype=tf.bool)
    ))
    def update(self, observation, action, reward, next_observation, done):
        """
        update: updates the parameters of the network
        issue: tf.function not working well
        """
        num_actions = len(observation)

        if done:
            """ if done, just add reward """
            target_Q = reward
        else:
            """ else, bootstrapping next Q value """
            target_Q = tf.add(reward, tf.scalar_mul(self.discount_factor, tf.reduce_max(self.bootstrap(next_observation))))

        with tf.GradientTape() as tape:
            action_one_hot = tf.one_hot(action, num_actions, dtype=tf.float64)
            responsible_Q = tf.reduce_sum(tf.multiply(self.call(observation), action_one_hot))
            # responsible_Q = self(tf.expand_dims(observation[action], 0))
            loss = tf.square(target_Q - responsible_Q)

        variables = self.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss
