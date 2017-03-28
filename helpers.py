import numpy as np
from scipy.io import wavfile
import glob
import pdb
import matplotlib.pyplot as plt

def get_learning_rate(t):
    learning_rate = 2e-9
    bounds = [1e5 * (2 ** i) for i in range(10)]
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

def load_weights(sess, analysis_ph, synthesis_ph):
    print ('loading weights')
    assign_analysis = analysis_ph.assign(np.load('saved/filters/analysis.npy'))
    assign_synthesis = synthesis_ph.assign(np.load('saved/filters/synthesis.npy'))
    sess.run(assign_analysis)
    sess.run(assign_synthesis)

def save_data(x_batch, x_hat_vals, analysis_vals, synthesis_vals):
    np.save('saved/filters/analysis.npy', analysis_vals)
    np.save('saved/filters/synthesis.npy', synthesis_vals)
    np.save('saved/samples/actual.npy', x_batch)
    np.save('saved/samples/reconstruction.npy', x_hat_vals)

class Plotter():
    def __init__(self, model):
        self.model = model

    def setup_plot_bf2(self):
        num_rows, num_cols = 16,8

        figure, axes = plt.subplots(num_rows, num_cols, figsize=(26,14))

        k = 0
        plots = []
        for i in range(num_rows):
            for j in range(num_cols):
                plots.append(axes[i,j].plot(np.zeros(self.model.n_input))[0])
                axes[i,j].set_ylim([-0.6,0.6])
                axes[i,j].xaxis.set_visible(False)
                axes[i,j].yaxis.set_visible(False)
                k = k + 1
        self.figure = figure
        self.plots = plots
        plt.show(block=False)

    def update_plot_bf2(self, synthesis):
        n_input = synthesis.shape[0]
        for k in range(len(self.plots)):
            self.plots[k].set_data(range(n_input), synthesis[:,k])
        self.figure.canvas.draw()
