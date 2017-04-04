import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import FastICA
import pdb
from optparse import OptionParser
from helpers import *

np.set_printoptions(threshold=np.nan)

parser = OptionParser()
parser.add_option("-s", "--source", dest="source", default="grid",
                  help="(pitt, environment, mammals, mix)")
(opt, args) = parser.parse_args()

N = 16000
sz = 128
nfilters = 64
plot_dim = np.ceil(np.sqrt(nfilters))
X = construct_data(opt.source, N, sz)

ica = FastICA(n_components=nfilters, whiten=True)
#ica = FastICA(n_components=nfilters)
ica.fit(X)
filters = ica.components_

#print ('Saving')
#np.save('dict/%s' % opt.source, filters)

#plotHeight = np.max(np.abs(filters ** 2))
plotHeight = np.max(np.abs(filters))
plt.figure()
plt.suptitle("Source: %s" % opt.source, fontsize=24)
for i in range(nfilters):
    axes = plt.subplot(plot_dim, plot_dim, i + 1)
    plt.plot(filters[i,:])
    #plt.axis([-plotHeight, plotHeight, 0, sz])
    plt.axis([0, sz, -plotHeight, plotHeight])
    axes.set_xticks([])
    axes.set_yticks([])

print ('Plotting...')
plt.show()

from numpy import fft
plt.figure()
plt.suptitle("Source: %s (Fourier)" % opt.source, fontsize=24)
for i in range(nfilters):
    axes = plt.subplot(plot_dim, plot_dim, i + 1)

    Fx = fft.fft(filters[i,:])
    freqs = fft.fftfreq(sz)
    plt.plot(freqs[sz//2:], np.abs(Fx)[sz//2:])

    #plt.plot(filters[i,:] ** 2)
    #plt.axis([-plotHeight, plotHeight, 0, sz])
    #plt.axis([0, sz, -plotHeight, plotHeight])
    #axes.set_xticks([])
    axes.set_yticks([])
print ('Plotting...')
plt.show()
