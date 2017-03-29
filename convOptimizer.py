import tensorflow as tf
import pdb
import numpy as np
from math import log
import matplotlib.pyplot as plt
from convAutoEncoder import ConvAutoEncoder
from helpers import *
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-l", "--load_filters", action='store_true', dest="load",
                  default=False)
(opt, args) = parser.parse_args()

class Model():
    n_input = 2 ** 10 # 2 * 14700
    n_filter_width = 128
    #n_filters = int(0.2 * 128) # Got to 7-8dB no sparsity
    n_filters = int(0.2 * 128)
    n_batch_size = 128
    n_runs = 2 ** 16
    Lambda = 0.0

model = Model()

n_input, n_filter_width, n_filters, n_batch_size, n_runs = model.n_input, model.n_filter_width, model.n_filters, model.n_batch_size, model.n_runs
auto_encoder = ConvAutoEncoder(model)

x_ph = tf.placeholder(tf.float32, shape=[n_batch_size, n_input, 1], name="input.data")
x_target_ph = tf.placeholder(tf.float32, shape=[n_batch_size, n_input, 1], name="x_target")

u_ph = auto_encoder.encode(x_ph)
x_hat_ph = auto_encoder.decode(u_ph)

#cost_op = tf.reduce_sum(tf.reduce_mean(tf.square(x_target_ph - x_hat_ph), axis=0)) + model.Lambda * tf.reduce_sum(tf.reduce_mean(tf.abs(u_ph), axis=0))
cost_op = tf.reduce_mean(tf.square(x_target_ph - x_hat_ph)) + \
            model.Lambda * tf.reduce_mean(tf.abs(u_ph))
init_op = tf.global_variables_initializer()

analysis_ph, synthesis_ph = auto_encoder.get_filters_ph()
norm_s_op = synthesis_ph.assign(tf.nn.l2_normalize(synthesis_ph, 0))

learning_rate_ph = tf.placeholder(tf.float32, shape=[])
optimizer = tf.train.GradientDescentOptimizer(learning_rate_ph).minimize(cost_op)

plot_bf = True # Use ipython optimizer.py --pylab=tk for remote host, ipython for local
with tf.Session() as sess:
    sess.run(init_op)
    if opt.load:
        load_weights_conv(sess, analysis_ph, synthesis_ph)

    noise = np.zeros((n_batch_size, n_filters))
    plotter = Plotter(model)
    if plot_bf:
        plotter.setup_plot()

    x_batch = np.zeros((n_batch_size, n_input))
    for t in range(model.n_runs):
        x_batch = construct_batch(n_input, n_filter_width, n_batch_size)

        feed_dict = {x_ph: x_batch, x_target_ph: x_batch, \
                     learning_rate_ph: get_learning_rate(t)}

        analysis_vals, synthesis_vals, u_vals, x_hat_vals, cost, _ = \
            sess.run([analysis_ph, synthesis_ph, u_ph, x_hat_ph, cost_op, optimizer], \
                feed_dict=feed_dict)

        sess.run(norm_s_op)

        if (t+1) % 25 == 0:
            save_data_conv(x_batch, x_hat_vals, analysis_vals, synthesis_vals)
            print ("Data saved | Mean(u)=%.2f" % (np.mean(np.abs(u_vals))))
        if plot_bf and t % 25 == 0:
            plotter.update_plot(synthesis_vals[:,:,0])
        if True or t % 5 == 0:
            print ("%d) Cost: %.3f, SNR: %.2fdB" % (t, cost, snr(x_batch, x_hat_vals)))
