import numpy as np
from scipy.io import wavfile
import glob
import pdb
import matplotlib.pyplot as plt

def snr(x_batch, x_hat_vals):
    R = x_batch - x_hat_vals

    var = x_batch.var().mean()
    mse = (R ** 2).mean()
    snr = 10 * np.log10(var/mse)
    return snr

def construct_batch(n_input, n_filter_width, n_batch_size):
    base = 'data/lewicki_audiodata'
    wav_files = glob.glob('%s/mammals/*.wav' % base)

    x_batch = np.zeros((n_batch_size, n_input, 1))
    for i in range(n_batch_size):
        wfile = np.random.choice(wav_files)
        Fs, x_raw = wavfile.read(wfile)
        start = np.random.randint(x_raw.shape[0] - n_input)
        x_batch[i,:,0] = x_raw[start:start+n_input]
    return x_batch

def construct_data(source, N, sz):
    X = np.zeros((N, sz))

    base = 'data/lewicki_audiodata'
    if source == "pitt":
        wav_files = glob.glob('%s/PittSounds/*.wav' % base)
    elif source == "environment":
        wav_files = glob.glob('%s/envsounds/*.wav' % base)
    elif source == "mammals":
        wav_files = glob.glob('%s/mammals/*.wav' % base)
    elif source == "mix":
        wf1 = glob.glob('%s/envsounds/*.wav' % base)
        wf2 = glob.glob('%s/mammals/*.wav' % base)
        ratio = int(np.ceil(2*len(wf2)/len(wf1))) # 2 to 1 (env to mammals)
        wav_files = wf1 * ratio + wf2
    elif source == "white":
        for i in range(N):
            X[i,:] = np.random.randn(sz)
        return X
    else:
        raise Exception("Unknown data source: %s" % source)

    perf = False
    for i in range(N):
        wfile = np.random.choice(wav_files)
        xFs, x_raw = wavfile.read(wfile)
        #x_raw = resample(x_raw, Fs) # Takes too long for now
        #print ("1", timer() - start) if perf else: pass

        start = np.random.randint(len(x_raw) - sz)
        X[i,:] = x_raw[start:start+sz]
    return X

def load_weights_impl(sess, analysis_ph, synthesis_ph, conv=True):
    print ('loading weights')
    base = 'saved/%sfilters' % ("conv_" if conv else "")
    assign_analysis = analysis_ph.assign(np.load('%s/analysis.npy' % base))
    assign_synthesis = synthesis_ph.assign(np.load('%s/synthesis.npy' % base))
    sess.run(assign_analysis)
    sess.run(assign_synthesis)

def load_weights(sess, analysis_ph, synthesis_ph):
    return load_weights_impl(sess, analysis_ph, synthesis_ph, False)

def load_weights_conv(sess, analysis_ph, synthesis_ph):
    return load_weights_impl(sess, analysis_ph, synthesis_ph)

def save_data_impl(x_batch, x_hat_vals, analysis_vals, synthesis_vals, conv=True):
    base_1 = 'saved/%sfilters' % ("conv_" if conv else "")
    base_2 = 'saved/%ssamples' % ("conv_" if conv else "")
    np.save('%s/analysis.npy' % base_1, analysis_vals)
    np.save('%s/synthesis.npy' % base_1, synthesis_vals)
    np.save('%s/actual.npy' % base_2, x_batch)
    np.save('%s/reconstruction.npy' % base_2, x_hat_vals)

def save_data(x_batch, x_hat_vals, analysis_vals, synthesis_vals):
    save_data_impl(x_batch, x_hat_vals, analysis_vals, synthesis_vals, False)

def save_data_conv(x_batch, x_hat_vals, analysis_vals, synthesis_vals):
    save_data_impl(x_batch, x_hat_vals, analysis_vals, synthesis_vals)

def get_learning_rate(t):
    learning_rate = 1e-9
    #learning_rate = 2e-13
    bounds = [1000 * (2 ** i) for i in range(10)]
    for bound in bounds:
        if t < bound:
            break
        learning_rate *= 0.5
        if t == bound:
            print ("Decreasing rate to: ", learning_rate)
    return learning_rate



class Plotter():
    def __init__(self, model):
        self.model = model

    def setup_plot(self):
        #num_rows, num_cols = 16,8
        num_rows, num_cols = 6,4

        #figure, axes = plt.subplots(num_rows, num_cols, figsize=(26,14))
        figure, axes = plt.subplots(num_rows, num_cols, figsize=(14,7))

        k = 0
        plots = []
        for i in range(num_rows):
            for j in range(num_cols):
                plots.append(axes[i,j].plot(np.zeros(self.model.n_filter_width))[0])
                axes[i,j].set_ylim([-0.6,0.6])
                axes[i,j].xaxis.set_visible(False)
                #axes[i,j].yaxis.set_visible(False)
                k = k + 1
        self.figure = figure
        self.plots = plots
        plt.show(block=False)

    def update_plot(self, synthesis):
        n_input = synthesis.shape[0]
        for k in range(synthesis.shape[1]):
            self.plots[k].set_data(range(n_input), synthesis[:,k])
        self.figure.canvas.draw()
        plt.show(block=False)
