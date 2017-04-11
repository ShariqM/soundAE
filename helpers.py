import numpy as np
from scipy.io import wavfile
import glob
import pdb
import matplotlib.pyplot as plt

def get_learning_rate_impl(t, start_rate, start_num_iters):
    learning_rate = start_rate
    bounds = [start_num_iters * (2 ** i) for i in range(10)]
    for bound in bounds:
        if t < bound:
            break
        learning_rate *= 0.5
        if t == bound:
            print ("Decreasing rate to: ", learning_rate)
    return learning_rate

def snr(x_batch, x_hat_vals):
    R = x_batch - x_hat_vals

    var = x_batch.var().mean()
    mse = (R ** 2).mean()
    snr = 10 * np.log10(var/mse)
    return snr

def get_wav_files(source):
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
    elif source == "mix+":
        wf1 = glob.glob('%s/envsounds/*.wav' % base)
        wf2 = glob.glob('%s/mammals/*.wav' % base)
        wf3 = glob.glob('%s/PittSounds/*.wav' % base)
        wav_files = wf1 + wf2 + wf3
    elif source == "grid":
        wav_files = glob.glob('data/wavs/s*.1/*.wav')
    else:
        raise Exception("Unknown data source: %s" % source)
    return wav_files

def construct_data(source, n_batch_size, n_input):
    divide_by = 100
    x_batch = np.zeros((n_batch_size, n_input))
    wav_files = get_wav_files(source)
    for i in range(n_batch_size):
        wfile = np.random.choice(wav_files)
        Fs, x_raw = wavfile.read(wfile)

        start = np.random.randint(len(x_raw) - n_input)
        x_batch[i,:] = x_raw[start:start+n_input] / divide_by
    return x_batch

def construct_conv_data(source, N, model):
    n_batch_size, n_input = model.n_batch_size, model.n_input
    x_data = np.zeros((N, n_batch_size, n_input, 1))
    for i in range(N):
        x_batch = construct_data(source, n_batch_size, n_input)
        x_data[i,:,:,:] = x_batch.reshape(n_batch_size, n_input, 1)
    return x_data

def construct_conv_batch(source, n_batch_size, n_input):
    x_batch = construct_data(source, n_batch_size, n_input)
    return x_batch.reshape(n_batch_size, n_input, 1)

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

class Plotter():
    def __init__(self, model):
        self.model = model
        self.figures = []
        self.plots = []

    def setup_plot(self):
        for t in range(2):
            n_rows, n_cols = self.model.n_rows_bf, self.model.n_cols_bf

            n_height, n_width = self.model.n_height_bf, self.model.n_width_bf
            figure, axes = plt.subplots(n_rows, n_cols, figsize=(n_width, n_height))
            #plt.title("Analysis" if t == 0 else "Synthesis")

            k = 0
            plots = []
            for i in range(n_rows):
                for j in range(n_cols):
                    plots.append(axes[i,j].plot(np.zeros(self.model.n_filter_width))[0])
                    if t == 0:
                        g = (50 / self.model.n_filter_width) * self.model.norm_factor
                        axes[i,j].set_ylim([-g,g])
                    else:
                        axes[i,j].set_ylim([-0.4,0.4])
                    axes[i,j].xaxis.set_visible(False)
                    axes[i,j].yaxis.set_visible(False)
                    k = k + 1
            plt.tight_layout()
            self.figures.append(figure)
            self.plots.append(plots)
            plt.show(block=False)

    def update_plot(self, analysis, synthesis, skip_synth=True):
        for t, filters in enumerate((analysis, synthesis)):
            if skip_synth and t > 0:
                break
            figure = self.figures[t]
            plots = self.plots[t]

            n_input = filters.shape[0]
            n_plots = min(len(plots), filters.shape[1])
            for k in range(n_plots):
                plots[k].set_data(range(n_input), filters[:,k])
            figure.canvas.draw()
            plt.show(block=False)

    def setup_plot_LIF(self):
        n_filters = self.model.n_filters
        n_input = self.model.n_input

        n_rows, n_cols = 4,4
        figure, axes = plt.subplots(n_rows, n_cols, figsize=(14,7))

        k = 0
        plots = []
        for j in range(n_cols):
            for i in range(n_rows):
                if i % 2 == 0:
                    axes[i,j].set_title("V")
                    axes[i,j].set_ylim([-1, self.model.threshold * 1.1])
                else:
                    axes[i,j].set_title("A")
                    axes[i,j].set_ylim([0,1.1])
                #axes[i,j].set_title("V" if i % 2 == 0 else "A")
                plots.append(axes[i,j].plot(np.zeros(n_input))[0])
                #axes[i,j].set_ylim([-0.4,0.4])
                #axes[i,j].xaxis.set_visible(False)
                #axes[i,j].yaxis.set_visible(False)
        self.figure = figure
        plt.show(block=False)
        self.plots = plots

    def update_plot_LIF(self, v_vals, a_vals):
        n_input = self.model.n_input
        j,k = 0,0
        n_plots = min(len(self.plots), v_vals.shape[1] * 2)
        for i in range(n_plots):
            if i % 2 == 0:
                self.plots[i].set_data(range(n_input), v_vals[:,k])
                j = j + 1
            else:
                self.plots[i].set_data(range(n_input), a_vals[:,k])
                k = k + 1
        self.figure.canvas.draw()
