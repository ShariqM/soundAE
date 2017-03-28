import tensorflow as tf
import pdb

def init_weights(n_1, n_2):
    return tf.truncated_normal((n_1, n_2), 0.0,
            1/tf.sqrt(tf.cast(n_1 + n_2, tf.float32)))

class AutoEncoder(object):
    def __init__(self, model):
        self.threshold = 0.1
        n_input, n_filters = model.n_input, model.n_filters
        self.A = tf.Variable(init_weights(n_filters, n_input), name="analysis_filters")
        self.S = tf.Variable(init_weights(n_input, n_filters), name="synthesis_filters")

    def get_filters_ph(self):
        return self.A, self.S

    def spike_activation_impl(self, x):
        cond = tf.less(x, tf.ones(tf.shape(x)) * self.threshold)
        out = tf.where(cond, tf.zeros(tf.shape(x)), tf.ones(tf.shape(x)))
        return out

    def spike_activation(self, x):
        return tf.sigmoid(x - self.threshold) + \
            tf.stop_gradient(self.spike_activation_impl(x) - tf.sigmoid(x - self.threshold))

    def thresh_relu(self, x):
        cond = tf.less(x, tf.ones(tf.shape(x)) * self.threshold)
        out = tf.where(cond, tf.zeros(tf.shape(x)), x - self.threshold)
        return out

    def encode(self, x_input, noise):
        u = tf.einsum("fi,bi->bf", self.A, x_input)
        r = u + noise
        return r
        #a = tf.sigmoid(r)
        #a = self.spike_activation(r)
        #a = self.thresh_relu(r)
        #a = tf.nn.relu(r)
        #return a

    def decode(self, u):
        x_hat = tf.einsum("if,bf->bi", self.S, u)
        return x_hat
