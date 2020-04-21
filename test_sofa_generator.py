# %%

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import itertools

import sys
import tools

# %%
block_size = 128
offset = None
filename = 'tmp/train/RIGID_SPHERE_anechoic.sofa'

# %%

gen = tools.preprocessing.SOFAAudioGenerator(
    [filename],
    noise=None,  # we use diffuse to simulate white sensor noise
    block_size=block_size,
    batch_size=360,
    # block_offset=offset,
    block_jitter=2,  # max +-2 samples
    SNRs=[30, 90],
    shuffle=False,
    mirror=False
)

# %%
_, xax = gen.y_func([0,0,0], 0)

for ii in range(len(gen)):
    x, y = gen[ii]

    fig, axs = plt.subplots(
        nrows=x.shape[2], 
        ncols=2, 
        sharey=False,
        figsize=(4, 4)
    )

    for kk in range(x.shape[2]):

        axs[kk,0].imshow(
            20*np.log10(np.abs(x[:,:,kk])),
            vmin=-100,
            vmax=-20,
            aspect="auto",
            origin='upper'
        )

        axs[kk,0].set_xlabel("time / samples")
        axs[kk,0].set_ylabel("scenario index")
        axs[kk,0].title.set_text('receiver index: {0}'.format(kk))
        axs[kk,0].grid(True)


    axs[0,1].imshow(
        np.transpose(y), 
        aspect="auto",
        extent=[xax[0],xax[-1],x.shape[0]-0.5,0-0.5],
        origin='upper'
    )
    # ax.plot(phi_mic[0], 0.05, 'k.')
    axs[0,1].set_xlabel("source azimuth")
    axs[0,1].grid(True)

    plt.show()

# %%
