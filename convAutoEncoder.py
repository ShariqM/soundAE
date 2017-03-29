import tensorflow as tf
import pdb

def init_encode_weights(n_filter_width, n_filters):
    return tf.truncated_normal((n_filter_width, 1, n_filters), 0.0,
            1/tf.sqrt(tf.cast(n_filter_width, tf.float32)))

def init_decode_weights(n_filter_width, n_filters):
    return tf.truncated_normal((n_filter_width, n_filters, 1), 0.0,
            1/tf.sqrt(tf.cast(n_filter_width, tf.float32)))

class ConvAutoEncoder(object):
    def __init__(self, model):
        self.threshold = 0.1
        n_filters, n_filter_width = model.n_filters, model.n_filter_width
        self.A = tf.Variable(init_encode_weights(n_filter_width, n_filters), name="analysis_filters")
        self.S = tf.Variable(init_decode_weights(n_filter_width, n_filters), name="synthesis_filters")


    def get_filters_ph(self):
        return self.A, self.S

    def encode(self, x_input):
        u = tf.nn.conv1d(x_input, self.A, 1, padding='SAME')
        return u

    def decode(self, u):
        x_hat = tf.nn.conv1d(u, self.S, 1, padding='SAME')
        return x_hat
