import tensorflow as tf
import pdb

def init_encode_weights(n_filter_width, n_filters):
    #return tf.zeros((n_filter_width, 1, n_filters))
    return tf.truncated_normal((n_filter_width, 1, n_filters), 0.0,
            1/tf.sqrt(tf.cast(n_filters + n_filter_width, tf.float32)))

def init_decode_weights(n_filter_width, n_filters):
    return tf.truncated_normal((n_filter_width, n_filters, 1), 0.0,
            1/tf.sqrt(tf.cast(n_filters + n_filter_width, tf.float32)))

class ConvAutoEncoder(object):
    def __init__(self, model):
        self.threshold = 0.1
        n_filters, n_filter_width = model.n_filters, model.n_filter_width
        self.A = tf.Variable(init_encode_weights(n_filter_width, n_filters), name="analysis_filters")
        self.S = tf.Variable(init_decode_weights(n_filter_width, n_filters), name="synthesis_filters")

    def get_filters_ph(self):
        return self.A, self.S

    def spike_activation_impl(self, x):
        cond = tf.less(x, tf.ones(tf.shape(x)) * self.threshold)
        out = tf.where(cond, tf.zeros(tf.shape(x)), tf.ones(tf.shape(x)))
        return out

    def spike_activation(self, x):
        return tf.sigmoid(x - self.threshold) + \
            tf.stop_gradient(self.spike_activation_impl(x) - tf.sigmoid(x - self.threshold))

    def encode(self, x_input, noise):
        u = tf.nn.conv1d(x_input, self.A, 1, padding='SAME')
        r = u + noise
        #u = tf.nn.relu(u)
        #u = self.spike_activation(u)
        return u, r

    def decode(self, r):
        x_hat = tf.nn.conv1d(r, self.S, 1, padding='SAME')
        return x_hat
