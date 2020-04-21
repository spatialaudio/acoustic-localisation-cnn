# %%
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import tools
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

block_size = 512
offset = 512

train_path = "tmp/train/"

dirsound = np.load(train_path + "anechoic_noise_E0_signals.npy")
dirlabels = np.load(train_path + "anechoic_noise_E0_azimuths.npy")
diffsound = np.load(train_path + "anechoic_diffuse_E0_signals.npy")

gen_random = tools.preprocessing.DirectionalAudioGenerator(
    dirsound,
    dirlabels,
    noise=diffsound,
    convert2rad=True,
    block_size=block_size,
    batch_size=256,
    block_offset=offset,
    SNRs=[0, 5, 10, 20, 30],
    shuffle=True,
    mirror=True,
    p_fb_ratio=0.75
)

model = tools.models.generate_correlation_model(
    block_size=gen_random.block_size,
    n_dirs=gen_random.p_steps,
    max_displacement=45,
    normalise=True,
    usefft=False
)

# %%
model.summary()
model.compile(
    optimizer='adam',
    loss=tools.losses.jensen_shannon_divergence
)

# %%
model.fit_generator(generator=gen_random, epochs=5)

# %%
model.save('tmp/models/correlation_dnn.h5')