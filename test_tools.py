# %%

import numpy as np
import tools
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.ndimage.interpolation as interp
import scipy.signal as _sig
import itertools

# %%

np.random.seed(42)

nsigs = 3
nsamples = 1024

noise = np.hstack((np.zeros(nsamples // 4), 
                   np.random.randn(nsamples // 2),
                   np.zeros(nsamples // 4)
                 ))
sig1 = np.zeros([noise.shape[0], nsigs])

wns = np.linspace(0.1, 0.9, nsigs)
for ii, wn in enumerate(wns):
    b, a = _sig.butter(8, 
                      [wn-0.05, wn+0.05], 
                      btype='bandpass', 
                      analog=False,
                      output='ba'
                     )
    sig1[:,ii] = _sig.lfilter(b,a, noise, axis=0)

sig2 = np.copy(sig1)
lags = np.linspace(-128, 128, num=nsigs, endpoint=True)
for ii, delta in enumerate(lags):
    sig2[:,ii] = interp.shift(sig1[:,ii], delta, cval=0.0)

tens1 = tf.convert_to_tensor(sig1, dtype=tf.float32)
tens2 = tf.convert_to_tensor(sig2, dtype=tf.float32)

tens1 = tf.expand_dims(tens1, axis=0)
tens2 = tf.expand_dims(tens2, axis=0)

tenslist = [tens1, tens2]

sess = tf.Session()
with sess.as_default():
    for md, usefft, normalise in itertools.product([128, 256, None],
                                                  [False, True], 
                                                  [False, True]
                                                 ):

        conv1d_fft = tools.layers.Corr1D(usefft=False, 
                                    max_displacement=md, 
                                    normalise=normalise,
                                    input_shape=[tens1.shape, tens2.shape])

        output = conv1d_fft(tenslist).eval()

        ax = plt.subplot()
        ax.plot(output[0,])
        plt.show()

#%%

Nps = 11
p_tails = np.linspace(0.0, 1.0, Nps)
p_tails = np.expand_dims(p_tails, axis=1)

y_pred = np.zeros((Nps, Nps, 2))
y_pred[:,:,0] = p_tails
y_pred[:,:,1] = 1 - p_tails

y_test = np.zeros((Nps, Nps, 2))
y_test[:,:,0] = np.transpose(p_tails)
y_test[:,:,1] = 1 - np.transpose(p_tails)

sess = tf.Session()
with sess.as_default():
    t_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    t_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

    loss = tools.losses.jensen_shannon_divergence(t_pred, t_test).eval()

fig = plt.figure()
plt.imshow(loss)
plt.colorbar()
