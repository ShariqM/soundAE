import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops

# See ~/anaconda3/envs/tf2/lib/python3.5/site-packages/tensorflow/python/ops/rnn_cell.py
class LIFCell(tf.contrib.rnn.RNNCell):
  """The LIF Cell"""

  def __init__(self, num_units, tau_RC=147):
    self.threshold = tau_RC * 0.9
    self.tau_RC = tau_RC
    self._num_units = num_units

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return 2 * self._num_units

  def spike_activation_impl(self, x):
    cond = tf.less(x, tf.ones(tf.shape(x)) * self.threshold)
    out = tf.where(cond, tf.zeros(tf.shape(x)), tf.ones(tf.shape(x)))
    return out

    ''' Use a sigmoid gradient instead of the non-differential sign func '''
  def spike_activation(self, x):
    with tf.variable_scope('spike_activation'):
      return tf.sigmoid(x - self.threshold) + \
        tf.stop_gradient(self.spike_activation_impl(x) - tf.sigmoid(x - self.threshold))

  def voltage_impl(self, x):
    cond = tf.less(x, tf.ones(tf.shape(x)) * self.threshold)
    out = tf.where(cond, x, tf.zeros(tf.shape(x)))
    return out

  def voltage_for_grad(self, x):
    cond = tf.less(x, tf.ones(tf.shape(x)) * self.threshold)
    out = tf.where(cond, x, self.threshold * tf.exp(-(x-self.threshold)))
    # FIXME RELU? XXX
    return out

  def voltage(self, x):
    return self.voltage_for_grad(x) + \
      tf.stop_gradient(self.voltage_impl(x) - self.voltage_for_grad(x))

  def __call__(self, inputs, state, scope=None):
    with vs.variable_scope(scope or type(self).__name__):
      v_bar_t = inputs + tf.exp(-1/self.tau_RC) * state
      #v_t = v_bar_t
      v_t = self.voltage(v_bar_t)
      a_t = self.spike_activation(v_bar_t)
      o_t = array_ops.concat([v_t, a_t], 1)
    return o_t, v_t
