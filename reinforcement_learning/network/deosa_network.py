import tensorflow as tf

from reinforcement_learning.network.network import Network

OBSERVATION_SIZE = 17


class DEOSANetwork(Network):
    """
     EDSSNetworkDQN network for variable size of action space

     Action (selection) is encoded as a vector and concatenated with user's state to form observation
     Q-value is calculated for each action and the service with the highest Q-value is selected

     Refer DRRN
     He, Ji, et al. "Deep reinforcement learning with an action space defined by natural language." (2016).
    """

    def __init__(self, learning_rate, discount_factor, tau, hidden_units, activation):
        super(DEOSANetwork, self).__init__(name="DEOSANetwork",
                                           learning_rate=learning_rate,
                                           discount_factor=discount_factor,
                                           tau=tau)

        self.input_layer = tf.keras.layers.InputLayer(input_shape=(None, OBSERVATION_SIZE), name="InputLayer")

        with tf.name_scope("HiddenLayers"):
            self.hidden_layers = [
                tf.keras.layers.Dense(
                    unit, activation=activation
                ) for unit in hidden_units
            ]
        # self.lstm_layer = tf.keras.layers.GRU(1024, activation=activation, name="RecurrentLayer")
        self.output_layer = tf.keras.layers.Dense(1, activation=None, name="OutputLayer")

        # self._set_inputs(inputs=self.input_layer)  # for saving model without compile or predict

        self.optimizer = tf.optimizers.Adam(learning_rate)

        # self.lstm_layer.build((None, None, hidden_units[-1]))
        self.build((None, None, OBSERVATION_SIZE))

    @tf.function(input_signature=(  # input_signature is specified to avoid frequent retracing
            tf.TensorSpec(shape=[None, None, OBSERVATION_SIZE], dtype=tf.float64),
    ))
    def call(self, observation, training=None, mask=None):
        z = self.input_layer(observation)
        for layer in self.hidden_layers:
            z = layer(z)
        # z = self.lstm_layer(z)
        output = self.output_layer(z)
        return output

    @tf.function(input_signature=(  # input_signature is specified to avoid frequent retracing
            tf.TensorSpec(shape=[None, None, OBSERVATION_SIZE], dtype=tf.float64),
            tf.TensorSpec(shape=[None], dtype=tf.int32),
            tf.TensorSpec(shape=[None], dtype=tf.float64),
            tf.TensorSpec(shape=[None, None, OBSERVATION_SIZE], dtype=tf.float64),
            tf.TensorSpec(shape=[None], dtype=tf.bool)
    ))
    def update(self, observation, action, reward, next_observation, done):
        """
        update: updates the parameters of the network
        issue: tf.function not working well
        """
        batch_size = len(observation)
        target_Q = tf.add(reward,
                          tf.squeeze(tf.multiply(self.discount_factor,
                                                 tf.reduce_max(self.bootstrap(next_observation),
                                                               axis=1)),
                                     axis=1))

        with tf.GradientTape() as tape:
            indices = tf.stack([tf.range(batch_size), action], axis=1)
            responsible_Q = tf.gather_nd(params=tf.squeeze(self.call(observation), axis=2),
                                         indices=indices,
                                         name="ResponsibleQ")

            loss = tf.reduce_sum(tf.square(tf.subtract(target_Q, responsible_Q)), name="Loss")

        variables = self.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss
