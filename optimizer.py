import tensorflow as tf
import pdb
import numpy as np
from math import log
import matplotlib.pyplot as plt
from autoEncoder import AutoEncoder
from helpers import *
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-l", "--load_filters", action='store_true', dest="load",
                  default=False)
(opt, args) = parser.parse_args()

class Model():
    n_input = 128
    n_filter_width = n_input
    n_filters = 128
    n_batch_size = 640
    n_runs = 2 ** 16
    Lambda = 4000.0

model = Model()

n_input, n_filters, n_batch_size, n_runs = model.n_input, model.n_filters, model.n_batch_size, model.n_runs
auto_encoder = AutoEncoder(model)

x_ph = tf.placeholder(tf.float32, shape=[n_batch_size, n_input], name="input.data")
n_ph = tf.placeholder(tf.float32, shape=[n_batch_size, n_filters],  name="white.noise")
x_target_ph = tf.placeholder(tf.float32, shape=[n_batch_size, n_input,], name="x_target")

u_ph = auto_encoder.encode(x_ph, n_ph)
x_hat_ph = auto_encoder.decode(u_ph)

cost_op = tf.reduce_sum(tf.reduce_mean(tf.square(x_target_ph - x_hat_ph), axis=0)) + model.Lambda * tf.reduce_sum(tf.reduce_mean(tf.abs(u_ph), axis=0))
init_op = tf.global_variables_initializer()

analysis_ph, synthesis_ph = auto_encoder.get_filters_ph()
norm_s_op = synthesis_ph.assign(tf.nn.l2_normalize(synthesis_ph, 0))

learning_rate_ph = tf.placeholder(tf.float32, shape=[])
optimizer = tf.train.GradientDescentOptimizer(learning_rate_ph).minimize(cost_op)

N = 4000
data = construct_data("mammals", N, n_input)

plot_bf  = True # Use ipython optimizer.py --pylab=tk for remote host, ipython for local
with tf.Session() as sess:
    sess.run(init_op)
    if opt.load:
        load_weights(sess, analysis_ph, synthesis_ph)

    noise = np.zeros((n_batch_size, n_filters))
    plotter = Plotter(model)
    if plot_bf:
        plotter.setup_plot()

    x_batch = np.zeros((n_batch_size, n_input))
    for t in range(model.n_runs):
        for i in range(n_batch_size):
            k = np.random.randint(data.shape[0])
            x_batch[i,:] = data[k,:]

        feed_dict = {x_ph: x_batch, n_ph: noise, x_target_ph: x_batch, \
                     learning_rate_ph: get_learning_rate(t)}

        analysis_vals, synthesis_vals, u_vals, x_hat_vals, cost, _ = \
            sess.run([analysis_ph, synthesis_ph, u_ph, x_hat_ph, cost_op, optimizer], \
                feed_dict=feed_dict)

        sess.run(norm_s_op)

        if (t+1) % 25 == 0:
            save_data(x_batch, x_hat_vals, analysis_vals, synthesis_vals)
            print ("Data saved | Mean(u)=%.2f" % (np.sum(np.mean(np.abs(u_vals), axis=0))))
        if plot_bf and t % 1000 == 0:
            plotter.update_plot(synthesis_vals)
        if t % 5 == 0:
            print ("%d) Cost: %.3f, SNR: %.2fdB" % (t, cost, snr(x_batch, x_hat_vals)))
