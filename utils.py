import tensorflow as tf


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def variable_summaries(variable_name, values, step):
    with tf.name_scope(variable_name):
        mean = tf.reduce_mean(values)
        tf.summary.scalar('mean', mean, step)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(values - mean)))
        tf.summary.scalar('stddev', stddev, step)
        tf.summary.scalar('max', tf.reduce_max(values), step)
        tf.summary.scalar('min', tf.reduce_min(values), step)
        # tf.summary.histogram('histogram', var)
