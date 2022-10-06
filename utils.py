import tensorflow as tf
import os
import errno

""" format of path that collects summary for each experiment """
SUMMARY_PATH = os.path.join(os.path.dirname(__file__), "summary")


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


def get_summary_path(agent, datetime, filename):
    """ get_summary_path: returns a file path for collecting summary for each experiment """
    file_path = os.path.join(SUMMARY_PATH, datetime, agent, filename)
    if not os.path.exists(os.path.dirname(file_path)):
        try:
            os.makedirs(os.path.dirname(file_path))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    return file_path
