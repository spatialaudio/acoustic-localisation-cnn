# %%
import numpy as np
import keras
import os
import tools
import matplotlib.pyplot as plt

#############################################
# https://github.com/dmlc/xgboost/issues/1715
# for my current Mac OSX
# might be not required on other machines
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
#############################################

model_file = 'tmp/models/sfa_cnn.h5'

custom_obj_dict = {
    'SplitLayer': tools.layers.SplitLayer,
    'Corr1D': tools.layers.Corr1D,
    'jensen_shannon_divergence': tools.losses.jensen_shannon_divergence
}

model = keras.models.load_model(model_file, custom_objects=custom_obj_dict)

# %%

train_path = "tmp/train/"

dirsound = _np.load(fileprefix + "noise_E0_signals.npy")
dirlabels = _np.load(fileprefix + "noise_E0_azimuths.npy")
diffsound = _np.load(fileprefix + "diffuse_E0_signals.npy")

gen = tools.preprocessing.DirectionalAudioGenerator(
    dirsound,
    dirlabels,
    noise=None,
    convert2rad=True,
    block_size=block_size,
    block_offset=offset,
    block_jitter=2,
    batch_size=None,
    SNRs=[0, 5, 10, 20],
    shuffle=False,
    mirror=False,
    p_fb_ratio=1.0
)

# %%

x_test, y_test = gen[45]

y_predict = model.predict(x_test)

phiplot = np.linspace(np.pi, -np.pi, 360)
phiplot = phiplot[::-1]

phimean_predict = np.angle(
    np.sum(np.mean(y_predict, axis=0) * np.exp(1j * phiplot)))
phimean_test = np.angle(np.sum(np.mean(y_test, axis=0) * np.exp(1j * phiplot)))

idx = 0
plt.plot(phiplot, np.mean(y_predict, axis=0))
plt.plot(phiplot, np.mean(y_test, axis=0))
plt.stem([phimean_predict], [0.025], "b")
plt.stem([phimean_test], [0.025], "r")
plt.show()

# %%
ndir = len(gen)

y_test_mean = np.zeros((ndir, 360))
y_predict_mean = np.zeros((ndir, 360))

for ii in range(len(gen)):
    x_test, y_test = gen[ii]
    y_predict = model.predict(x_test)
    y_test_mean[ii, :] = np.mean(y_test, axis=0)
    y_predict_mean[ii, :] = np.mean(y_predict, axis=0)

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 30))
axs[0].imshow(y_predict_mean)
axs[1].imshow(y_test_mean)
plt.show()
