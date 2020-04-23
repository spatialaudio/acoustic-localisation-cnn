# %%
from sklearn.model_selection import train_test_split
import argparse  # for command line
import multiprocessing
import glob  # searching for file patterns
import tensorflow as tf
import keras
import sys
import tools
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # do not use GPU

stdout = sys.stdout
sys.stdout = open('/dev/null', 'w')
sys.stdout = stdout

# deactivate Tensorflow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def main(args):

    if args.noise_file is not None:
        noise = np.transpose(np.load(args.noise_file))
    else:
        noise = None

    generator_params = {
        'block_size': 512,  # also defines input layer of DNN
        'block_offset': 256,
        'block_jitter': 1,
        'batch_size': 64,
        'snr': [0, 5, 10, 20],
        'reestimate_snr': True,
        'scale_range': None,
        'samplewise_center': True,
        'samplewise_std_normalisation': True,
        'noise': noise,
        'y_func': tools.preprocessing.p_dist_one_hot_uniform
    }

    epochs = 10

    # training and validation data
    file_list = glob.glob(args.data_pattern)
    train_file_list, valid_file_list = train_test_split(
        file_list, test_size=0.2)

    gen_train = tools.preprocessing.SOFAAudioGenerator(
        train_file_list,
        shuffle=True,
        mirror=False,
        **generator_params
    )

    gen_valid = tools.preprocessing.SOFAAudioGenerator(
        valid_file_list,
        shuffle=False,
        mirror=False,
        **generator_params
    )

    # Load checkpoint:
    if args.checkpoint_pattern is not None:
        model = tools.io.load_newest_model(args.checkpoint_pattern)
    else:
        # Init model:
        model = tools.models.generate_equivalent_cnn_2d(
            block_size=gen_train.block_size,
            n_channels=gen_train.n_channels,
            n_dirs=gen_train.y_size
        )

    if args.verbose:
        model.summary()

    model.compile(
        optimizer='adam',
        loss=keras.losses.kullback_leibler_divergence,
        metrics=[keras.losses.kullback_leibler_divergence,
                 tools.losses.jensen_shannon_divergence
                 ]
    )

    # create callback for checkpoints after each epoch
    checkpoint_path = args.model_prefix + \
        "-checkpoint-{epoch:04d}-{val_loss:.4f}.h5"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=args.verbose,
        save_weights_only=False,
        monitor='val_loss'
    )

    # actual training
    model.fit_generator(
        generator=gen_train,
        validation_data=gen_valid,
        verbose=args.verbose,
        workers=args.jobs,
        epochs=epochs,
        max_queue_size=128,
        callbacks=[checkpoint_callback],
        use_multiprocessing=True,  # without it, segmentation fault due to HDF5
        initial_epoch=args.initial_epoch
    )

    model.save(args.model_prefix + '-final.h5')


# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Main script for running training of CNN'
    )
    parser.add_argument(
        '--data_pattern', '-d',
        type=str,
        default='tmp/train/*_noise*.sofa',
        help='Pattern used by glob to find files for training (default: %(default)s)'
    )
    # Todo: support use of multiple datasets using command line. currently only supported when editing default params
    parser.add_argument(
        '--model_prefix', '-m',
        type=str,
        default='tmp/models/model',
        help='file prefix for saving checkpoints and final model. (default: %(default)s)'
    )
    parser.add_argument(
        '--noise_file', '-n',
        type=str,
        default=None,
        help='noise added to stimuli. (default: %(default)s)'
    )
    parser.add_argument(
        '--checkpoint_pattern', '-c',
        type=str,
        default=None,
        help='Provide pattern of checkpoint model file to load - newest file selected (default: %(default)s)'
    )
    parser.add_argument(
        '--initial_epoch', '-i',
        type=int,
        default=0,
        help='Epoch to start from when loading from checkpoint. (default: %(default)s)'
    )
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=10,
        help='Maximum number of epochs (default: %(default)s)'
    )
    parser.add_argument(
        '--jobs', '-j',
        type=int,
        default=multiprocessing.cpu_count(),
        help='Number of parallel jobs used for training (default: %(default)s)'
    )
    parser.add_argument(
        '--verbose', '-v',
        default=False,
        action='store_true',
        help='Verbose mode (default: %(default)s)'
    )

    args = parser.parse_args()

    main(args)
