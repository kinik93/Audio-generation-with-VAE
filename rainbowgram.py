import librosa
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['svg.fonttype'] = 'none'
from scipy.io.wavfile import read as readwav

cdict  = {'red':  ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'green': ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'blue':  ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'alpha':  ((0.0, 1.0, 1.0),
                   (1.0, 0.0, 0.0))
        }

my_mask = matplotlib.colors.LinearSegmentedColormap('MyMask', cdict)
plt.register_cmap(cmap=my_mask)

# sr, audio = readwav('guitar_acoustic_015-067-050.wav')
# audio = audio.astype(np.float32)


def view_rainbowgram(path):

    audio, sr = librosa.load(path, sr=16000)

    C = librosa.cqt(audio, sr, hop_length=256,
                    bins_per_octave=40, n_bins=240,
                    filter_scale=0.8,
                    fmin=librosa.note_to_hz('C2')
                    )

    mag = np.abs(C)
    phase_angle = np.angle(C)
    phase_unwrapped = np.unwrap(phase_angle) #Unwrap by changing deltas between values to 2*pi complement.
    dphase =  phase_unwrapped[:, 1:] - phase_unwrapped[:, :-1]
    dphase = np.concatenate([phase_unwrapped[:, 0:1], dphase], axis=1) / np.pi
    mag = (librosa.core.power_to_db(mag**2, amin=1e-13, top_db=70.0, ref=np.max)/70.0) + 1

    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)
    fig.subplots_adjust(left=0.1, right=0.9, wspace=0.05, hspace=0.1)
    axes.matshow(dphase[::-1, :], cmap=plt.cm.rainbow) #invert row order of dphase
    axes.matshow(mag[::-1, :], cmap=my_mask)

    axes.set_yticklabels([])
    axes.set_xticklabels([])

    plt.show()

def save_rainbowgram(source_path, dest_path):

    audio, sr = librosa.load(source_path, sr=16000)

    C = librosa.cqt(audio, sr, hop_length=256,
                    bins_per_octave=40, n_bins=240,
                    filter_scale=0.8,
                    fmin=librosa.note_to_hz('C2')
                    )

    mag = np.abs(C)
    phase_angle = np.angle(C)
    phase_unwrapped = np.unwrap(phase_angle) #Unwrap by changing deltas between values to 2*pi complement.
    dphase =  phase_unwrapped[:, 1:] - phase_unwrapped[:, :-1]
    dphase = np.concatenate([phase_unwrapped[:, 0:1], dphase], axis=1) / np.pi
    mag = (librosa.core.power_to_db(mag**2, amin=1e-13, top_db=70.0, ref=np.max)/70.0) + 1

    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)
    fig.subplots_adjust(left=0.1, right=0.9, wspace=0.05, hspace=0.1)
    axes.matshow(dphase[::-1, :], cmap=plt.cm.rainbow) #invert row order of dphase
    axes.matshow(mag[::-1, :], cmap=my_mask)

    axes.set_yticklabels([])
    axes.set_xticklabels([])

    plt.savefig(dest_path)