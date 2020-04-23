# %%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # do not use GPU
import numpy as np
import keras
import tools
import matplotlib.pyplot as plt
import glob  # searching for file patterns
import multiprocessing
import argparse
import tqdm

# %%
def main(args):

    if args.noise_file is not None:
        noise = np.transpose(np.load(args.noise_file))
    else:
        noise = None

    custom_obj_dict = {
        'SplitLayer': tools.layers.SplitLayer,
        'Corr1D': tools.layers.Corr1D,
        'jensen_shannon_divergence': tools.losses.jensen_shannon_divergence
    }

    model = tools.io.load_newest_model(args.model_pattern)

    generator_params = {
        'block_size': model.input_shape[1],  # also defines input layer of DNN
        'block_offset': model.input_shape[1] // 2,
        'block_jitter': 0,
        'batch_size': None,
        'snr': [np.Inf],
        'reestimate_snr': False,
        'shuffle': False,
        'mirror': False,
        'scale_range': None,
        'samplewise_center': True,
        'samplewise_std_normalisation': True,
        'noise': noise,
        'y_func': tools.preprocessing.p_dist_one_hot_uniform
    }

    # testing data
    test_file_list = glob.glob(args.data_pattern)

    # data generator
    gen_test = tools.preprocessing.SOFAAudioGenerator(
        test_file_list,
        **generator_params
    )

    # actual testing
    y_predict = model.predict_generator(
        generator=gen_test,
        verbose=args.verbose,
        workers=args.jobs,
        max_queue_size=128,
        use_multiprocessing=True,  # without it, segmentation fault due to HDF5
    )

    nbatches = len(gen_test)
    batch_size = gen_test.batch_size
    y_predict_avg = np.zeros((nbatches, gen_test.y_size))
    for bdx in range(nbatches):
        y_predict_avg[bdx, :] = np.mean(
            y_predict[bdx*batch_size:(bdx+1)*batch_size, :],
            axis=0,
            keepdims=True
        )

    plt.imshow(y_predict_avg, aspect='auto')
    plt.savefig('plot.png')
    # plt.show()

    result = model.evaluate_generator(
        generator=gen_test,
        verbose=args.verbose,
        workers=args.jobs,
        max_queue_size=128,
        use_multiprocessing=True,  # without it, segmentation fault due to HDF5
    )


# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Main script for running training of CNN'
    )
    parser.add_argument(
        '--data_pattern', '-d',
        type=str,
        default='tmp/test/*_noise*.sofa',
        help='Pattern used by glob to find files for testing (default: %(default)s)'
    )
    # Todo: support use of multiple datasets using command line. currently only supported when editing default params
    parser.add_argument(
        '--model_pattern', '-m',
        type=str,
        default='tmp/models/model-final.h5',
        help='Pattern of model file to load - newest file selected (default: %(default)s)'
    )
    parser.add_argument(
        '--noise_file', '-n',
        type=str,
        default=None,
        help='noise added to stimuli. (default: %(default)s)'
    )
    parser.add_argument(
        '--jobs', '-j',
        type=int,
        default=multiprocessing.cpu_count(),
        help='Number of parallel jobs used for testing (default: %(default)s)'
    )
    parser.add_argument(
        '--verbose', '-v',
        default=False,
        action='store_true',
        help='Verbose mode (default: %(default)s)'
    )

    args = parser.parse_args()

    main(args)
