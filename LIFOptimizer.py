import tensorflow as tf
from tensorflow.python.ops import array_ops
import pdb
import numpy as np
from math import log
import matplotlib.pyplot as plt
from convAutoEncoder import ConvAutoEncoder
from helpers import *
from optparse import OptionParser
from LIFCell import LIFCell

parser = OptionParser()
parser.add_option("-l", "--load_filters", action='store_true', dest="load",
                  default=False)
parser.add_option("-v", "--visualizer", action='store_true', dest="plot_bf",
                  default=False)
(opt, args) = parser.parse_args()
#opt.load = True
#opt.plot_bf = True

class Model():
    n_input = 2 ** 10
    n_steps = n_input
    n_filter_width = 2 ** 4
    #n_filters = int(0.2 * 128) # Got to 7-8dB with linear, no sparsity
    #n_filters = 128
    n_filters = 2 ** 3
    n_batch_size = 2 ** 5
    n_runs = 2 ** 16
    Lambda = 0000.0

model = Model()

n_input, n_filter_width, n_filters, n_batch_size, n_runs = \
    model.n_input, model.n_filter_width, model.n_filters, model.n_batch_size, model.n_runs
n_steps = n_input

#x_ph = tf.placeholder(tf.float32, shape=[n_batch_size, n_input, 1], name="input.data")
#x_target_ph = tf.placeholder(tf.float32, shape=[n_batch_size, n_input, 1], name="x_target")

#u_ph = auto_encoder.encode(x_ph)

u_ph = tf.placeholder(tf.float32, shape=[n_batch_size, n_steps, n_filters], name="input.data")
cell_ph = LIFCell(n_filters)
output_ph, _ = tf.nn.dynamic_rnn(cell_ph, u_ph, dtype=tf.float32)
v_ph, a_ph = array_ops.split(output_ph, 2, axis=2)

#x_hat_ph = auto_encoder.decode(a_ph)
#cost_op = tf.reduce_mean(tf.square(x_target_ph - x_hat_ph)) + \

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)

    for t in range(model.n_runs):
        u_batch = np.ones((n_batch_size, n_steps, n_filters))

        feed_dict = {u_ph: u_batch}
        v_vals, a_vals = sess.run([v_ph, a_ph], feed_dict)

        # Plot
        figure, axes = plt.subplots(2,1)
        axes[0].plot(v_vals[0,:,0])
        axes[1].plot(a_vals[0,:,0])
        plt.show()
