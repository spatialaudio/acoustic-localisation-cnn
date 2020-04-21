# %%
import numpy as np
import keras
import tools
import matplotlib.pyplot as plt
import os
import glob  # searching for file patterns
import multiprocessing
import argparse
import tqdm

# %%
def main(args):

    custom_obj_dict = {
        'SplitLayer': tools.layers.SplitLayer,
        'Corr1D': tools.layers.Corr1D,
        'jensen_shannon_divergence': tools.losses.jensen_shannon_divergence
    }

    model = keras.models.load_model(
        args.model_file,
        custom_objects=custom_obj_dict
    )

    generator_params = {
        'block_size': model.input_shape[1],  # also defines input layer of DNN
        'block_offset': None,
        'block_jitter': 0,
        'batch_size': None,
        'snr': [0, 5, 10, 20],
        'reestimate_snr': False,
        'shuffle': False,
        'mirror': True,
        'scale_range': None,
        'samplewise_center': True,
        'samplewise_std_normalisation': True
    }

    # testing data
    test_file_list = glob.glob(args.data_pattern)

    # data generator
    gen_test = tools.preprocessing.SOFAAudioGenerator(
        test_file_list,
        **generator_params
    )


    # actual testing
    # y_predict = model.predict_generator(
    #     generator=gen_test,
    #     verbose=args.verbose,
    #     workers=args.jobs,
    #     max_queue_size=128,
    #     use_multiprocessing=True,  # without it, segmentation fault due to HDF5
    # )

    # plt.imshow(y_predict, aspect='auto')
    # plt.show()

    # 
    # result = model.evaluate_generator(
    #     generator=gen_test,
    #     verbose=args.verbose,
    #     workers=args.jobs,
    #     max_queue_size=128,
    #     use_multiprocessing=True,  # without it, segmentation fault due to HDF5
    # )

    # print(result)


    # 
    ndir = len(gen_test)
    y_test_mean = np.zeros((ndir, gen_test.y_size))
    y_predict_mean = np.zeros((ndir,  gen_test.y_size))

    for ii in tqdm.trange(ndir):
        x_test, y_test = gen_test[ii]
        y_predict = model.predict(x_test)
        y_test_mean[ii, :] = np.mean(y_test, axis=0)
        y_predict_mean[ii, :] = np.mean(y_predict, axis=0)

    fig, axs = plt.subplots(nrows=1, ncols=2)
    axs[0].imshow(y_predict_mean, aspect='auto')
    axs[1].imshow(y_test_mean, aspect='auto')
    plt.show()


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
        '--model_file', '-m',
        type=str,
        default='tmp/models/model-final.h5',
        help='file of model. (default: %(default)s)'
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