from scipy.signal import butter, lfilter
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz
from scipy.io import wavfile
import glob

def butter_bandpass(lowcut, highcut, Fs, order=5):
    nyq = 0.5 * Fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, Fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, Fs, order=order)
    y = lfilter(b, a, data)
    return y

# Sample rate and desired cutoff frequencies (in Hz).
Fs = 25000.0
lowcut = 100.0 # 250=100Hz | 6000=60 * 250 = 2500 * 6 = 15000
highcut = 10000.0

i = 0
wav_files = glob.glob('data/wavs/s1/*.wav')
for wav_file in wav_files:
    Fs, x = wavfile.read(wav_file)
    x_new = butter_bandpass_filter(x, lowcut, highcut, Fs, order=6)
    fold = wav_file.split('/')[-2] + '.1'
    name = wav_file.split('/')[-1]
    #print ('data/wavs/s1.1/%s' % name, Fs, x.astype(np.int16))
    wavfile.write('data/wavs/%s/%s' % (fold, name), Fs, x.astype(np.int16))
