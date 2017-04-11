import tensorflow as tf
import pdb
import numpy as np
from math import log
import matplotlib.pyplot as plt
from convAutoEncoder import ConvAutoEncoder
from helpers import *
from optparse import OptionParser
import datetime
start = datetime.datetime.now()

parser = OptionParser()
parser.add_option("-l", "--load_filters", action='store_true', dest="load",
                  default=False)
parser.add_option("-v", "--visualizer", action='store_true', dest="plot_bf",
                  default=False)
parser.add_option("-d", "--data_source", dest="data",
                  default="mix+", help="Data source (mix, grid)")
(opt, args) = parser.parse_args()
#opt.load = False
#opt.plot_bf = False

class Model():
    n_input = 2 ** 12
    n_filter_width = 128
    n_filters = 128
    n_batch_size = 16
    n_runs = 2 ** 16
    Lambda = 16000

    start_rate = 4e-7
    start_num_iters = 107
    adamOptimizer = False

    neuron_entropy = 2.5 # units of bits/sample
    kill_noise = False

    p = [4, 6, 5, 12]
    #p = [8, 8, 10, 14]
    n_rows_bf = p[0]
    n_cols_bf = p[1]
    n_height_bf = p[2]
    n_width_bf = p[3]

    # Somehow making this smaller leads to better reconstructions everything else fixed....
    #norm_factor = 0.025
    u_var = 2
    norm_factor = 0.1

def get_learning_rate(t):
    return get_learning_rate_impl(t, model.start_rate, model.start_num_iters)

model = Model()

n_input, n_filter_width, n_filters, n_batch_size, n_runs = \
    model.n_input, model.n_filter_width, model.n_filters, model.n_batch_size, model.n_runs
auto_encoder = ConvAutoEncoder(model)
# Cost = MSE + L1

x_ph = tf.placeholder(tf.float32, shape=[n_batch_size, n_input, 1], name="input.data")
n_ph = tf.placeholder(tf.float32, shape=[n_batch_size, n_input, n_filters], name="noise")
x_target_ph = tf.placeholder(tf.float32, shape=[n_batch_size, n_input, 1], name="x_target")

u_ph, r_ph = auto_encoder.encode(x_ph, n_ph)
x_hat_ph = auto_encoder.decode(r_ph)

#cost_op = tf.reduce_sum(tf.reduce_mean(tf.square(x_target_ph - x_hat_ph), axis=0)) + model.Lambda * tf.reduce_sum(tf.reduce_mean(tf.abs(u_ph), axis=0))
cost_op = tf.reduce_mean(tf.square(x_target_ph - x_hat_ph)) + \
            model.Lambda * tf.reduce_mean(tf.log(1 + u_ph ** 2))
#cost_op = tf.reduce_mean(tf.square(x_target_ph - x_hat_ph)) + \
            #model.Lambda * tf.reduce_mean(tf.abs(u_ph))

analysis_ph, synthesis_ph = auto_encoder.get_filters_ph()
norm_a_op = analysis_ph.assign(tf.nn.l2_normalize(analysis_ph, 0) * model.norm_factor)

learning_rate_ph = tf.placeholder(tf.float32, shape=[])
if model.adamOptimizer:
    train_op = tf.train.AdamOptimizer(model.start_rate).minimize(cost_op)
else:
    #train_op = tf.train.GradientDescentOptimizer(learning_rate_ph).minimize(cost_op)
    train_op = tf.train.MomentumOptimizer(learning_rate_ph, 0.9).minimize(cost_op)
grad_op  = tf.train.GradientDescentOptimizer(learning_rate_ph).compute_gradients(cost_op)


N = 1000
x_data = construct_conv_data(opt.data, N, model)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    if opt.load:
        load_weights_conv(sess, analysis_ph, synthesis_ph)
    sess.run(norm_a_op)

    identity_filter = tf.ones((n_filter_width, 1, 1))
    x_energy_ph = tf.nn.conv1d(x_ph, identity_filter, 1, padding='SAME')
    x_energy = np.zeros((N, n_batch_size, n_input, 1))
    for i in range(N):
        x_energy[i,:,:,:] = sess.run(x_energy_ph, feed_dict={x_ph: np.abs(x_data[i,:,:,:])})
        x_energy[i,:,:,:] /= n_filter_width

    plotter = Plotter(model)
    if opt.plot_bf:
        plotter.setup_plot()

    denom = 2 ** (2 * model.neuron_entropy) - 1
    x_batch = np.zeros((n_batch_size, n_input))
    noise_batch = np.zeros((n_batch_size, n_input, n_filters))
    for t in range(model.n_runs):
        idx = np.random.randint(N)
        x_batch = x_data[idx,:,:,:]
        x_ebatch = x_energy[idx,:,:,:]
        #x_batch = construct_conv_batch(opt.data, n_batch_size, n_input)
        if not model.kill_noise:
            #noise_batch = np.random.normal(0, np.sqrt(model.u_var/denom), (n_batch_size, n_input, n_filters))
            for b in range(n_batch_size):
                for i in range(n_input):
                    noise_batch[b,i,:] = np.random.normal(0, x_ebatch[b,i,0] / denom, (n_filters))

        feed_dict = {x_ph: x_batch, n_ph: noise_batch, x_target_ph: x_batch, \
                     learning_rate_ph: get_learning_rate(t)}

        u_vals, r_vals, x_hat_vals, cost, grad_vals, _ = \
            sess.run([u_ph, r_ph, x_hat_ph, cost_op, grad_op, train_op], \
                feed_dict=feed_dict)

        uu_var = np.mean(u_vals ** 2)

        mess = False
        if mess:
            fig, axes = plt.subplots(5,1)
            axes[0].plot(x_batch[0,:,0])
            axes[1].plot(np.mean(u_vals[0,:,:] ** 2, axis=1))
            axes[2].plot(np.mean(r_vals[0,:,:] ** 2, axis=1))
            axes[3].plot(np.mean(noise_batch[0,:,:], axis=1))
            axes[4].plot(np.mean((u_vals[0,:,:] + noise_batch[0,:,:]) ** 2, axis=1))
            plt.show()
            #print ("u_var",
            print (u_vals.shape)
            plt.figure()
            print ('hist')
            #plt.hist(u_vals.flatten() ** 2, bins='auto')
            plt.hist(u_vals[:,200,:].flatten() ** 2, bins=300)
            plt.show()

            #figure, ax = plt.subplots(2,1)
            #ax[0].imshow(u_vals[0,:,:].T, interpolation='none')
            #ax[0].set_aspect('auto') # Fill y-axis
            #ax[1].plot(x_batch[0,:,0])
            #plt.show(block=True)

        analysis_grad = grad_vals[0][0]
        synthesis_grad = grad_vals[1][0]
        sess.run(norm_a_op)
        analysis_vals, synthesis_vals = sess.run([analysis_ph, synthesis_ph])
        #model.start_rate = 0.1/(np.max(np.abs(analysis_grad)))
        #print ("Rate: ", model.start_rate)

        if t > 0 and t % 25 == 0:
            save_data_conv(x_batch, x_hat_vals, analysis_vals, synthesis_vals)
            print ("Data saved")
        if opt.plot_bf and t % 10 == 0:
            print ("Updating")
            plotter.update_plot(analysis_vals[:,0,:], synthesis_vals[:,:,0])
        if t % 5 == 0:
            elapsed = (datetime.datetime.now() - start).seconds
            #mean_u = np.mean(np.abs(u_vals))
            mean_u = np.mean(np.log(1 + u_vals ** 2))
            print ("%d) T=%ds Cost: %.3f, SNR: %.2fdB, Mean(u)=%.2f, U_VAR: %.2f" % \
                (t, elapsed, cost, snr(x_batch, x_hat_vals), mean_u, uu_var))
