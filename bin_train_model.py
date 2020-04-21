# %%
import numpy as np
import tools
import keras
import tensorflow as tf
import os
import glob  # searching for file patterns
import multiprocessing
import argparse

from sklearn.model_selection import train_test_split

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # do not use GPU
print("Num GPUs Available: ",
      len(tf.config.experimental.list_physical_devices('GPU')))

# deactivate Tensorflow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def main(args):

    generator_params = {
        'block_size': 512,  # also defines input layer of DNN
        'block_offset': 256,
        'block_jitter': 2,
        'batch_size': 32,
        'snr': [0, 5, 10, 20],
        'reestimate_snr': True,
        'scale_range': None,
        'samplewise_center': True,
        'samplewise_std_normalisation': True
    }

    epochs = 10

    # training and validation data
    file_list = glob.glob(args.data_pattern)
    train_file_list, valid_file_list = train_test_split(
        file_list, test_size=0.2)

    gen_train = tools.preprocessing.SOFAAudioGenerator(
        train_file_list,
        shuffle=True,
        mirror=True,
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
        # possible custom loss functions and layers
        custom_obj_dict = {
            'SplitLayer': tools.layers.SplitLayer,
            'Corr1D': tools.layers.Corr1D,
            'jensen_shannon_divergence': tools.losses.jensen_shannon_divergence
        }

        latest_file = max(
            glob.glob(args.checkpoint_pattern),  # all files matching pattern
            key=os.path.getctime  # time of last modification
        )
        # Load model:
        model = keras.models.load_model(
            latest_file,
            custom_objects=custom_obj_dict
        )
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
        loss=tools.losses.jensen_shannon_divergence
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
        '--checkpoint_pattern', '-c',
        type=str,
        default=None,
        help='Option to provide pattern of checkpoint to load (newest file selected)'
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
