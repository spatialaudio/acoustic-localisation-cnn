# %%
import numpy as np
import matplotlib.pyplot as plt

n_blocks = 2**16
block_size = 16

SNR = -20
gamma = 10 ** (SNR / 10.0)
w = 1.0 / np.sqrt(1.0 + 1.0/gamma)

sig = w * np.random.randn(1, n_blocks, block_size)
phis = 0.0
ns = np.array([np.cos(phis), np.sin(phis)], ndmin=3)
ns = np.transpose(ns, axes=(2, 1, 0))

n_dirs = 360
phin = np.linspace(-np.pi, np.pi, n_dirs, endpoint=False)
nn = np.stack((np.cos(phin), np.sin(phin)), axis=1)
noise = np.random.randn(n_dirs, 1, n_blocks, block_size)

iinstant = (np.mean(noise, axis=0) + sig) * \
    (np.mean(noise * nn[:,:,np.newaxis,np.newaxis], axis=0) + sig * ns)

iavg = np.mean(iinstant, axis=2)

phiavg = np.arctan2(iavg[1, ], iavg[0, ])

h = np.histogram(phiavg, bins=360, range=(-np.pi, np.pi))

# h = plt.hist(phiavg, range=(-np.pi, np.pi), bins=100)

plt.plot(h[0] / np.sum(h[0]))
plt.show()

