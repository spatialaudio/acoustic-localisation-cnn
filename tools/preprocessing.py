import keras as _keras
import numpy as _np
import sofa as _sofa
import soundfile as _sf
import itertools as _iter
import abc as _abc


class SpatialAudioGenerator(_keras.utils.Sequence, _abc.ABC):
    """Meta class for audio data generation"""

    def __init__(self,
                 noise=None,
                 snr=[_np.Inf],
                 batch_size=None,
                 block_size=512,
                 block_offset=None,
                 block_jitter=0,
                 mirror=False,
                 shuffle=True,
                 y_func=None,
                 reestimate_snr=False,
                 scale_range=None,
                 samplewise_center=False,
                 samplewise_std_normalisation=False
                 ):

        super(_abc.ABC, self).__init__()

        # noise files
        self.noise = noise

        # Block related attributes
        self.block_size = block_size
        self.block_offset = block_size if block_offset is None else block_offset
        self.block_jitter = block_jitter

        # SNR related attributes
        self.gamma = 10 ** (_np.array(snr) / 10.0)  # linear SNR
        self.w = 1.0 / _np.sqrt(1.0 + 1.0 / self.gamma)
        self.num_snr = int(self.w.size)

        self.mirror = mirror
        self.shuffle = shuffle

        # audio data related attribute
        self.reestimate_snr = reestimate_snr
        self.scale_range = scale_range
        self.samplewise_center = samplewise_center
        self.samplewise_std_normalisation = samplewise_std_normalisation

        # label related attributes
        if y_func is None:
            self.y_func = p_dist_von_mises
        else:
            self.y_func = y_func
        _tmp, _ = self.y_func([0.0, 0.0, 0.0], 0.01)
        self.y_size = _tmp.size  # get size of labels

        # some numbers
        self.n_channels = 0
        self.n_scenarios = 0
        self.n_frames = 0
        self.n_samples = 0
        self.batch_size = batch_size

    def __len__(self):
        """Denotes the number of batches per epoch."""

        return int(_np.floor(self.n_samples / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data."""

        # Generate indexes of the batch
        indexes = self.indexes[index *
                               self.batch_size:(index + 1) * self.batch_size]

        # order indexes to speed up access
        indexes.sort()

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        self.indexes = _np.arange(self.n_samples)
        if self.shuffle:
            _np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        """Generates data containing batch_size samples."""
        # Initialization
        X = _np.empty(
            (self.batch_size, self.block_size, self.n_channels)
        )
        y = _np.empty((self.batch_size, self.y_size))

        # Generate data
        for ii, ID in enumerate(indexes):

            #
            mir_idx, snr_idx, data_idx = _np.unravel_index(
                ID, (self.mirror + 1, self.num_snr, self.n_frames)
            )

            # get data and position
            X[ii, ], pos = self.get_data_slice(data_idx)

            # current SNR
            gamma = self.gamma[snr_idx]

            # check if noise should be added
            if self.w[snr_idx] < 1.0:
                # check if custom noise was provided
                if self.noise is None:
                    # generate standard normal noise on each channel
                    noise_block = _np.random.randn(
                        self.block_size, self.n_channels
                    )
                else:
                    # randomly pick block from provided noise
                    ndx = _np.random.randint(
                        0, self.noise.shape[0] - self.block_size + 1
                    )
                    noise_block = self.noise[ndx:ndx + self.block_size]

                # re-estimate SNR
                if self.reestimate_snr:
                    sigma2_S = _np.mean(X[ii, ]**2)
                    sigma2_N = _np.mean(noise_block**2)
                    gamma *= sigma2_S / sigma2_N

                X[ii, ] *= self.w[snr_idx]
                X[ii, ] += _np.sqrt(1 - self.w[snr_idx]**2) * noise_block

            # scaling
            if self.scale_range is not None:
                X[ii, ] *= _np.random.uniform(*self.scale_range)

            # centering
            if self.samplewise_center:
                X[ii, ] -= _np.mean(X[ii, ])

            # std-normalisation
            if self.samplewise_std_normalisation:
                X[ii, ] *= 1.0 / _np.maximum(_np.std(X[ii, ]), 1e-6)

            # channels are either indexed normally and in reverse order
            mir_sign = -1 if mir_idx else 1

            # reverse order, if necessary
            X[ii, ] = X[ii, :, ::mir_sign]

            # Store labels
            pos[0] = mir_sign*pos[0]
            y[ii, ], _ = self.y_func(pos, gamma)

        return X, y

    @_abc.abstractmethod
    def get_data_slice(self, data_index):
        pass


class NumpyAudioGenerator(SpatialAudioGenerator):
    """Generates data for Keras."""

    def __init__(self,
                 signal,
                 positions,
                 convert2rad=False,
                 **kwargs
                 ):
        """Initialization."""

        super().__init__(**kwargs)

        self.signal = signal
        self.positions = _np.deg2rad(
            positions) if convert2rad else positions

        self.n_blocks = int(
            _np.floor((self.signal.shape[0] - self.block_size) / self.block_offset) + 1)

        self.n_channels = self.signal.shape[1]
        self.n_scenarios = self.signal.shape[2]
        self.n_frames = self.n_blocks*self.n_scenarios
        self.n_samples = (self.mirror + 1) * self.num_snr * self.n_frames

        if self.batch_size is None:
            self.batch_size = self.n_blocks

        self.on_epoch_end()

    def get_data_slice(self, data_index):

        sce_idx, blk_idx = _np.unravel_index(
            data_index, (self.n_scenarios, self.n_blocks)
        )

        # get signal blocks
        sig_idx = blk_idx * self.block_offset
        if self.block_jitter > 0:
            jitter = _np.random.randint(
                -self.block_jitter, high=self.block_jitter+1, size=self.n_channels
            )

            X = _np.empty((self.block_size, self.n_channels))
            for jj, shift in enumerate(jitter):
                adx = _np.maximum(0, sig_idx + shift)
                adx = _np.minimum(
                    adx, self.signal.shape[0] - self.block_size)
                X[:, jj] = self.signal[adx:adx +
                                       self.block_size, jj, sce_idx]
        else:
            X = self.signal[sig_idx:sig_idx+self.block_size, :, sce_idx]

        y = self.positions[:, sce_idx]

        return X, y


class SOFAAudioGenerator(SpatialAudioGenerator):
    """Generates data for Keras."""

    def __init__(self,
                 ir_files,
                 **kwargs
                 ):
        """Initialization."""

        super().__init__(**kwargs)

        #
        self.ir_files = ir_files

        self.n_channels = None
        self.n_scenarios = 0
        self.n_frames = 0
        self.fmeb = _np.empty((0, 4), dtype=int)

        self._cur_fdx = None  # currently opened file index
        self._cur_ir = None  # currently opened file handle

        # Read some metadata
        for fdx, ir_file in enumerate(self.ir_files):

            ir = _sofa.Database.open(ir_file)

            # handle number of receivers
            if self.n_channels is None:
                self.n_channels = ir.Dimensions.R
            elif not (self.n_channels == ir.Dimensions.R):
                raise RuntimeError(
                    'IR files do not have same number of receivers')

            Nblocks = int(
                _np.floor((ir.Dimensions.N - self.block_size) /
                          self.block_offset) + 1
            )

            edx = _np.arange(ir.Dimensions.E)
            mdx = _np.arange(ir.Dimensions.M)
            bdx = _np.arange(Nblocks)

            _tmp_indexes = _np.stack(
                _np.meshgrid(fdx, edx, mdx, bdx), -1
            ).reshape(-1, 4)

            self.fmeb = _np.vstack((self.fmeb, _tmp_indexes))
            self.n_scenarios += ir.Dimensions.E*ir.Dimensions.M
            self.n_frames += Nblocks*ir.Dimensions.E*ir.Dimensions.M

            ir.close()

        self.n_samples = self.n_frames * (self.mirror + 1) * self.num_snr

        if self.batch_size is None:
            # set to average number of blocks
            self.batch_size = int(
                _np.floor(self.n_frames / self.n_scenarios)
            )

        self.on_epoch_end()

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        super().on_epoch_end()

        if self._cur_ir is not None:
            self._cur_ir.close()

    def get_data_slice(self, data_index):
        # file, emitter, measurement, and block index
        fil_idx, emi_idx, mea_idx, blk_idx = tuple(self.fmeb[data_index])

        if self._cur_fdx != fil_idx:
            if not (self._cur_fdx is None):
                self._cur_ir.close()
            self._cur_ir = _sofa.Database.open(self.ir_files[fil_idx])
            self._cur_fdx = fil_idx

        X = _np.empty((self.block_size, self.n_channels))
        # get signal blocks
        sig_idx = blk_idx * self.block_offset
        if self.block_jitter > 0:
            jitter = _np.random.randint(
                -self.block_jitter,
                high=self.block_jitter+1,
                size=self.n_channels
            )
            for cha_idx, shift in enumerate(jitter):
                adx = _np.maximum(0, sig_idx + shift)
                adx = _np.minimum(
                    adx, self._cur_ir.Dimensions.N - self.block_size)

                X[:, cha_idx] = self._cur_ir.Data.IR.get_values(
                    indices={
                        "E": emi_idx,
                        "M": mea_idx,
                        "R": cha_idx,
                        "N": slice(adx, adx + self.block_size)
                    }
                )
        else:
            X = self._cur_ir.Data.IR.get_values(
                indices={
                    "E": emi_idx,
                    "M": mea_idx,
                    "N": slice(sig_idx, sig_idx + self.block_size)
                },
                dim_order=('N', 'R')
            )

        # get labels
        y = self._cur_ir.Emitter.Position.get_relative_values(
            self._cur_ir.Listener,
            indices={
                "E": emi_idx,
                "R": 0,
                "M": mea_idx,
            },
            system=_sofa.spatial.System.Spherical,
            angle_unit='rad'
        )

        return X, y


def p_dist_von_mises(pos, gamma, bins=360, kappa_max=100.0):
    kappa = _np.minimum(2 * gamma, kappa_max)
    phi = _np.linspace(_np.pi, -_np.pi, endpoint=False, num=bins)
    p = _np.exp(kappa * _np.cos(phi[::-1] - pos[0]))

    return p / _np.sum(p), phi[::-1]


def p_dist_one_hot_uniform(pos, gamma, bins=360):

    phi = _np.linspace(_np.pi, -_np.pi, endpoint=False, num=bins)
    phi = phi[::-1]
    delta = _np.mod(pos[0] - phi, 2*_np.pi)
    idx = _np.argmax(_np.maximum(
        _np.abs(delta), _np.abs(2*_np.pi - delta)))

    # weight = 1.0 / (1.0 + 1.0 / gamma)
    weight = 1.0

    p = _np.ones_like(phi) * (1.0 - weight) / bins
    p[idx] += weight

    return p, phi
