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
parser.add_option("-d", "--data_source", dest="data",
                  default="mix+", help="Data source (mix, grid)")
parser.add_option("-l", "--load_filters", action='store_true', dest="load",
                  default=False)
parser.add_option("-f", "--visualze_filters", action='store_true', dest="plot_bf",
                  default=False)
parser.add_option("-v", "--visualize_LIF", action='store_true', dest="plot_LIF",
                  default=False)
(opt, args) = parser.parse_args()
#opt.load = True
#opt.plot_bf = True
#opt.plot_LIF = True

class Model():
    n_input = 2 ** 8
    n_filter_width = 16 # 256 / 16 = 16
    n_filters = 4
    n_batch_size = 2
    n_runs = 2 ** 16
    Lambda = 1.0

    tau_RC = n_filter_width / 2
    threshold = 10

def get_learning_rate(t):
    start_rate = 3e-1
    start_num_iters = 100
    return get_learning_rate_impl(t, start_rate, start_num_iters)

model = Model()

# Parameters
n_input, n_filter_width, n_filters, n_batch_size, n_runs = \
    model.n_input, model.n_filter_width, model.n_filters, model.n_batch_size, model.n_runs
#n_steps = n_input

# Input, Output
x_ph = tf.placeholder(tf.float32, shape=[n_batch_size, n_input, 1], name="input.data")
n_ph = tf.placeholder(tf.float32, shape=[n_batch_size, n_input, n_filters], name="noise")
x_target_ph = tf.placeholder(tf.float32, shape=[n_batch_size, n_input, 1], name="x_target")

# Network
auto_encoder = ConvAutoEncoder(model)

# Encode
u_ph, r_ph = auto_encoder.encode(x_ph, n_ph)
# LIF
cell_ph = LIFCell(n_filters, model)
output_ph, _ = tf.nn.dynamic_rnn(cell_ph, r_ph, dtype=tf.float32)
v_ph, a_ph = array_ops.split(output_ph, 2, axis=2)
# Decode
x_hat_ph = auto_encoder.decode(a_ph)

# Cost, Optimization
cost_op = tf.reduce_mean(tf.square(x_target_ph - x_hat_ph)) + \
            model.Lambda * tf.reduce_mean(tf.abs(r_ph))
learning_rate_ph = tf.placeholder(tf.float32, shape=[])
optimizer = tf.train.GradientDescentOptimizer(learning_rate_ph).minimize(cost_op)
grad_op  = tf.train.GradientDescentOptimizer(learning_rate_ph).compute_gradients(cost_op)

# ops
init_op = tf.global_variables_initializer()
analysis_ph, synthesis_ph = auto_encoder.get_filters_ph()

print ("Foo")
N = 1000
#x_data = construct_conv_data(opt.data, N, model)
print ("Foo")

def sin_wave():
    f = 16
    signal = np.zeros(model.n_input)
    for i in range(1,3):
        signal += np.sin(np.linspace(0, f*np.pi, model.n_input))
    return signal

with tf.Session() as sess:
    sess.run(init_op)
    if opt.load:
        load_weights_conv(sess, analysis_ph, synthesis_ph)

    print ("Setup plot")
    plotter = Plotter(model)
    if opt.plot_bf:
        plotter.setup_plot()
    if opt.plot_LIF:
        plotter.setup_plot_LIF_3(sin_wave())
        #figure, axes = plt.subplots(1,1)
        #p1 = axes.plot(np.zeros(n_input))[0]
        #p2 = axes.plot(np.zeros(n_input))[0]
        #axes.set_ylim([-1.1, 1.1])
        #plotter.setup_plot_LIF()
    print ("Setup for runs")

    noise_batch = np.zeros((n_batch_size, n_input, n_filters))
    x_batch = np.zeros((n_batch_size, n_input, 1))
    x_batch[0,:,0] = sin_wave()
    x_batch[1,:,0] = sin_wave()
    for t in range(model.n_runs):
        #idx = np.random.randint(N)
        #x_batch = x_data[idx,:,:,:]
        #x_batch = sin_wave()

        feed_dict = {x_ph: x_batch, x_target_ph: x_batch, n_ph: noise_batch, \
                     learning_rate_ph: get_learning_rate(t)}

        v_vals, a_vals, x_hat_vals, cost, grad_val,  _ = \
            sess.run([v_ph, a_ph, x_hat_ph, cost_op, grad_op, optimizer], feed_dict)

        analysis_vals, synthesis_vals = sess.run([analysis_ph, synthesis_ph])

        analysis_grad = grad_val[0][0]
        synthesis_grad = grad_val[1][0]

        print ('Mean Grad Val:', np.mean((np.abs(grad_val[0][0]))))
        if opt.plot_LIF:
            #plotter.update_plot_LIF_2(v_vals, a_vals, x_hat_vals, analysis_vals, synthesis_vals)
            plotter.update_plot_LIF_3(x_hat_vals[0,:,0], a_vals[0,:,:], v_vals[0,:,:],
                            analysis_vals[:,0,0], synthesis_vals[:,0,0])
            #p1.set_data(range(n_input), x_batch[0,:,0])
            #p2.set_data(range(n_input), x_hat_vals[0,:,0])
            #figure.canvas.draw()
            #plotter.update_plot_LIF(v_vals[0,:,:], a_vals[0,:,:])

        # Norm ?
        #if (t+1) % 25 == 0:
            #save_data_conv(x_batch, x_hat_vals, analysis_vals, synthesis_vals)
            #print ("Data saved | Mean(u)=%.2f" % (np.mean(np.abs(u_vals))))
        if opt.plot_bf and t % 500 == 0:
            plotter.update_plot(synthesis_vals[:,:,0])
        if True or t % 5 == 0:
            print ("%d) Cost: %.3f, SNR: %.2fdB" % (t, cost, snr(x_batch, x_hat_vals)))
        if t > 50 or snr(x_batch, x_hat_vals) > 8:
          print ("SHOW")
          plt.show(block=True)
