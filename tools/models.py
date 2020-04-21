import sofa as _sofa
import numpy as _np
import scipy.signal as _sig
import keras
import soundfile as sf
from . import layers as _layers


def generatelocalisationmodel(block_size, n_channels=2, n_dirs=360):

    Nmic = 4

    # single channel CNN
    single = keras.models.Sequential()
    single.add(keras.layers.Conv1D(filters=Nmic, kernel_size=4,
                                   padding='same', activation='relu'))
    single.add(keras.layers.MaxPooling1D(pool_size=2))
    single.add(keras.layers.Conv1D(filters=Nmic, kernel_size=4,
                                   padding='same', activation='relu'))
    single.add(keras.layers.MaxPooling1D(pool_size=2))
    single.add(keras.layers.Conv1D(filters=Nmic, kernel_size=4,
                                   padding='same', activation='relu'))
    single.add(keras.layers.MaxPooling1D(pool_size=2))

    input = keras.layers.Input((block_size, n_channels))
    splitouts = _layers.SplitLayer()(input)  # layer to split channels

    # apply (same) CNN to each channel
    cnnouts = [single(s) for s in splitouts]

    # joint processing
    out = keras.layers.Concatenate()(cnnouts)  # merge channels again
    out = keras.layers.Flatten()(out)
    out = keras.layers.Dense(8*Nmic, activation='sigmoid')(out)
    out = keras.layers.Dropout(0.25)(out)
    out = keras.layers.Dense(8*Nmic, activation='sigmoid')(out)
    out = keras.layers.Dense(n_dirs, activation='softmax')(out)

    return keras.models.Model(input, out)


def generate_correlation_model(block_size,
                               n_dirs=360,
                               max_displacement=None,
                               normalise=True,
                               usefft=False
                               ):

    # input layer
    input = keras.layers.Input((block_size, 2))

    # split channels
    splitouts = _layers.SplitLayer()(input)  # layer to split channels

    # single channel CNN
    single = keras.models.Sequential()
    single.add(keras.layers.Conv1D(filters=8, kernel_size=16,
                                   padding='same', activation='relu'))

    cnnout1 = single(splitouts[0])
    cnnout2 = single(splitouts[1])

    # merge each filter output by correlation
    out = _layers.Corr1D(usefft=usefft,
                         max_displacement=max_displacement,
                         normalise=normalise
                         )([cnnout1, cnnout2])
    # necessary since Dense layers are applied to last dimension
    out = keras.layers.Permute((2, 1))(out)

    out = keras.layers.Dense(32, activation='sigmoid')(out)
    out = keras.layers.Dense(32, activation='sigmoid')(out)
    out = keras.layers.Dense(n_dirs, activation='softmax')(out)

    out = keras.layers.Permute((2, 1))(out)  # undo Permute

    splitouts = _layers.SplitLayer()(out)

    out = keras.layers.Average()(splitouts)
    out = keras.layers.Flatten()(out)

    return keras.models.Model(input, out)


def generate_equivalent_cnn_2d(
    block_size,
    n_channels=2,
    n_dirs=360,
    n_filters=16,
    n_dense=32
):

    model = keras.models.Sequential()
    model.add(
        keras.layers.Reshape(
            (block_size, n_channels, 1),
            input_shape=(block_size, n_channels)
        )
    )
    model.add(keras.layers.BatchNormalization())

    # convolution layers
    for _ in range(3):
        model.add(
            keras.layers.Conv2D(
                filters=n_filters,
                kernel_size=(4, 1),
                padding='same',
            )
        )
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 1)))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Flatten())

    # dense layers
    for _ in range(2):
        model.add(keras.layers.Dense(n_dense))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Dropout(0.25))

    # output layer
    model.add(keras.layers.Dense(n_dirs, activation='softmax'))

    return model


def generate_cnn_2d(block_size, n_channels=2, n_dirs=360):

    model = keras.models.Sequential()
    model.add(
        keras.layers.Reshape(
            (block_size, n_channels, 1),
            input_shape=(block_size, n_channels)
        )
    )

    model.add(keras.layers.Conv2D(
        filters=5, kernel_size=(3, 3), padding='valid', activation='relu')
    )
    model.add(keras.layers.BatchNormalization(
        epsilon=1e-06, mode=0, momentum=0.9, weights=None))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 1), strides=(2, 1)))
    model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Conv2D(
        filters=5, kernel_size=(3, 3), padding='valid', activation='relu')
    )
    model.add(keras.layers.BatchNormalization(
        epsilon=1e-06, mode=0, momentum=0.9, weights=None))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 1), strides=(2, 1)))
    model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Conv2D(
        filters=5, kernel_size=(3, 3), padding='valid', activation='relu')
    )
    model.add(keras.layers.BatchNormalization(
        epsilon=1e-06, mode=0, momentum=0.9, weights=None))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 1), strides=(2, 1)))
    model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(16, activation='sigmoid'))
    model.add(keras.layers.Dense(16, activation='sigmoid'))
    model.add(keras.layers.Dense(n_dirs, activation='softmax'))

    return model


def evaluate_model(model, generator, eval_func=None):

    if eval_func is None:
        def eval_func(x): return _np.mean(x, axis=0)

    ndir = len(generator)

    y_test_mean = _np.zeros((ndir, generator.p_phi))
    y_predict_mean = _np.zeros((ndir, 360))

    for ii in range(len(generator)):
        x_test, y_test = generator[ii]
        y_predict = model.predict(x_test)

        y_test_mean[ii, :] = _np.mean(y_test, axis=0)
        y_predict_mean[ii, :] = _np.mean(y_predict, axis=0)

    return (y_test_mean, y_predict_mean)
