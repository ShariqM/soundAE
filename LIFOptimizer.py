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
parser.add_option("-f", "--visualze_filters", action='store_true', dest="plot_bf",
                  default=False)
parser.add_option("-v", "--visualize_LIF", action='store_true', dest="plot_LIF",
                  default=False)
(opt, args) = parser.parse_args()
#opt.load = True
#opt.plot_bf = True
opt.plot_LIF = True

class Model():
    n_input = 2 ** 8
    n_filter_width = 32
    n_filters = 8
    n_batch_size = 32
    n_runs = 2 ** 16
    Lambda = 0000.0

    tau_RC = 147
    threshold = 30

def get_learning_rate(t):
    start_rate = 1e-2
    start_num_iters = 100
    return get_learning_rate_impl(t, start_rate, start_num_iters)

model = Model()

print ("Model")
# Parameters
n_input, n_filter_width, n_filters, n_batch_size, n_runs = \
    model.n_input, model.n_filter_width, model.n_filters, model.n_batch_size, model.n_runs
#n_steps = n_input

# Input, Output
x_ph = tf.placeholder(tf.float32, shape=[n_batch_size, n_input, 1], name="input.data")
x_target_ph = tf.placeholder(tf.float32, shape=[n_batch_size, n_input, 1], name="x_target")

# Network
auto_encoder = ConvAutoEncoder(model)

# Encode
u_ph = auto_encoder.encode(x_ph)
# LIF
print ("LIF")
cell_ph = LIFCell(n_filters, model)
output_ph, _ = tf.nn.dynamic_rnn(cell_ph, u_ph, dtype=tf.float32)
v_ph, a_ph = array_ops.split(output_ph, 2, axis=2)
# Decode
x_hat_ph = auto_encoder.decode(a_ph)

# Cost, Optimizatoin
cost_op = tf.reduce_mean(tf.square(x_target_ph - x_hat_ph)) + \
            model.Lambda * tf.reduce_mean(tf.abs(u_ph))
learning_rate_ph = tf.placeholder(tf.float32, shape=[])
optimizer = tf.train.GradientDescentOptimizer(learning_rate_ph).minimize(cost_op)

# ops
init_op = tf.global_variables_initializer()
analysis_ph, synthesis_ph = auto_encoder.get_filters_ph()

with tf.Session() as sess:
    sess.run(init_op)
    if opt.load:
        load_weights_conv(sess, analysis_ph, synthesis_ph)

    print ("Setup plot")
    plotter = Plotter(model)
    if opt.plot_bf:
        plotter.setup_plot()
    if opt.plot_LIF:
        figure, axes = plt.subplots(1,1)
        p1 = axes.plot(np.zeros(n_input))[0]
        p2 = axes.plot(np.zeros(n_input))[0]
        axes.set_ylim([-1.1, 1.1])
        plotter.setup_plot_LIF()
    print ("Setup for runs")

    x_batch = np.zeros((n_batch_size, n_input))
    for t in range(model.n_runs):
        x_batch = construct_batch(n_input, n_filter_width, n_batch_size, norm=True)

        feed_dict = {x_ph: x_batch, x_target_ph: x_batch, \
                     learning_rate_ph: get_learning_rate(t)}

        v_vals, a_vals, x_hat_vals, cost, _ = \
            sess.run([v_ph, a_ph, x_hat_ph, cost_op, optimizer], feed_dict)

        if opt.plot_LIF:
            p1.set_data(range(n_input), x_batch[0,:,0])
            p2.set_data(range(n_input), x_hat_vals[0,:,0])
            figure.canvas.draw()
            plotter.update_plot_LIF(v_vals[0,:,:], a_vals[0,:,:])

        #analysis_vals, synthesis_vals, u_vals, x_hat_vals, cost, _ = \
            #sess.run([analysis_ph, synthesis_ph, u_ph, x_hat_ph, cost_op, optimizer], \
                #feed_dict=feed_dict)

        # Norm ?
        #if (t+1) % 25 == 0:
            #save_data_conv(x_batch, x_hat_vals, analysis_vals, synthesis_vals)
            #print ("Data saved | Mean(u)=%.2f" % (np.mean(np.abs(u_vals))))
        if opt.plot_bf and t % 50 == 0:
            plotter.update_plot(synthesis_vals[:,:,0])
        if True or t % 5 == 0:
            print ("%d) Cost: %.3f, SNR: %.2fdB" % (t, cost, snr(x_batch, x_hat_vals)))
