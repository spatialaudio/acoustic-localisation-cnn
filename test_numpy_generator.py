# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import sys
sys.path.append("..")
import tools

# np.random.seed(3)

# %%
block_size = 64
offset = 33
n_dirs = 8
n_channels = 36
n_sig = 128
n_noise = 128  # has to be >= n_sig

c = 343
fs = 48000
R = 0.1  # array dimension

# %%
# Test scenario: free-field, plane wave, circular open array

# bandlimited impulse (FIR filter, windowing method)
impulse = np.zeros(n_sig)
impulse[n_sig//2] = 1
# for even N -> linphase FIR type 2, for odd N -> linphase FIR type 1
# -60 dB highest stopband ripple
if True:
    impulse = sig.firwin(numtaps=n_sig, cutoff=fs * 0.3125, fs=fs,
                            window=('kaiser', 0.1102 * (60 - 8.7)))

signal, pos, _ = tools.audio.pressure_plane_wave_open_sphere(
    impulse, R, n_channels, n_dirs, fs, c)

# %%

gen = tools.preprocessing.NumpyAudioGenerator(
    signal,
    pos,
    noise=None,  # we use diffuse to simulate white sensor noise
    block_size=block_size,
    batch_size=7,
    block_offset=offset,
    block_jitter=2,  # max +-2 samples @ 48 kHz
    SNRs=[30, 90],
    shuffle=False,
    mirror=False
)

# %%
_, xax = gen.y_func([0,0,0], 0)

for ii in range(len(gen)):
    x, y = gen[ii]

    fig = plt.figure(figsize=(8, 16))

    for jj in range(gen.batch_size):
        ax = plt.subplot(gen.batch_size, 2, 2 * jj + 1)
        ax.imshow(20 * np.log10(abs(np.transpose(x[jj, ]))),
                  vmin=-60, vmax=0, aspect='auto', cmap='magma'
                  )
        ax.set_xlabel("time / samples")
        ax.set_ylabel("sensor index")
        ax.grid(True)

        ax = plt.subplot(gen.batch_size, 2, 2 * jj + 2)
        ax.plot(xax, np.transpose(y[jj]))
        # ax.plot(phi_mic[0], 0.05, 'k.')
        ax.set_ylim([0, 0.1])
        ax.set_xlabel("source azimuth")
        ax.set_ylabel("probability density")
        ax.grid(True)

    plt.show()

# %%
