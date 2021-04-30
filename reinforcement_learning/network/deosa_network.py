import tensorflow as tf

from reinforcement_learning.network.network import Network


class DQNetwork(Network):
    """
     EDSSNetworkDQN network for variable size of action space

     Action (selection) is encoded as a vector and concatenated with user's state to form observation
     Q-value is calculated for each action and the service with the highest Q-value is selected

     Refer DRRN
     He, Ji, et al. "Deep reinforcement learning with an action space defined by natural language." (2016).
    """

    def target(self, observations, actions, rewards, next_observations, done):
        return tf.add(rewards,
                      tf.squeeze(tf.multiply(self.discount_factor,
                                             tf.reduce_max(self.bootstrap(next_observations),
                                                           axis=1)),
                                 axis=1))


class SoftDQNetwork(DQNetwork):
    def target(self, observations, actions, rewards, next_observations, done):
        bootstrap = tf.squeeze(self.bootstrap(next_observations), axis=2)
        stochastic_policy = tf.nn.softmax(tf.divide(bootstrap, self.temperature), axis=1)

        maximum_entropy = tf.multiply(self.temperature, tf.math.log(stochastic_policy))

        return tf.add(rewards,
                      tf.multiply(
                          self.discount_factor,
                          tf.reduce_sum(
                              tf.multiply(
                                  stochastic_policy,
                                  tf.subtract(
                                      bootstrap,
                                      maximum_entropy
                                  )
                              )
                              , axis=1)
                      ))


class MunchausenDQNetwork(DQNetwork):
    def target(self, observations, actions, rewards, next_observations, done):
        batch_size = len(observations)

        bootstrap = tf.squeeze(self.bootstrap(next_observations), axis=2)
        stochastic_policy = tf.nn.softmax(tf.divide(bootstrap, self.temperature), axis=1)

        indices = tf.stack([tf.range(batch_size), actions], axis=1)
        current_policy = tf.gather_nd(
            params=tf.nn.softmax(tf.divide(tf.squeeze(self.bootstrap(observations), axis=2), self.temperature), axis=1),
            indices=indices,
            name="CurrentQ")
        munchausen = tf.multiply(self.alpha, tf.multiply(self.temperature, tf.math.log(current_policy)))

        maximum_entropy = tf.multiply(self.temperature, tf.math.log(stochastic_policy))

        return tf.add(tf.add(rewards, munchausen),
                      tf.multiply(
                          self.discount_factor,
                          tf.reduce_sum(
                              tf.multiply(
                                  stochastic_policy,
                                  tf.subtract(
                                      bootstrap,
                                      maximum_entropy
                                  )
                              )
                              , axis=1)
                      ))
