import argparse
import multiprocessing as mp
import os
import tools
import tqdm


def spatialise_noise_helper(arg_dict):
    tools.audio.spatialise_signal(**arg_dict)


def main(args):

    train_path = os.path.join(args.directory, "train")
    test_path = os.path.join(args.directory, "test")

    train_ir_remote = [
        # 'https://zenodo.org/record/55418/files/QU_KEMAR_anechoic_2m.sofa'
    ]

    test_ir_remote = [
        'https://zenodo.org/record/160751/files/QU_KEMAR_spirit.sofa'
    ]

    ir_local = tools.io.sync_remote_filelist(train_ir_remote, train_path)
    ir_local += tools.io.sync_remote_filelist(test_ir_remote, test_path)

    # create output filesnames and parameters
    param_list = []
    for filename in ir_local:
        filename_wo_ending = filename.rsplit('.', 1)[0]

        for ndx in range(args.num_noise):
            out_file = filename_wo_ending + '_dirnoise{0}.sofa'.format(ndx)

            param_list.append(
                {
                    'ir_file': filename,
                    'output_file': out_file,
                    'noise_length': args.length
                }
            )

    # multicore-processing
    with mp.Pool(args.jobs) as p:
        with tqdm.tqdm(total=len(param_list)) as pbar:
            for _ in p.imap_unordered(spatialise_noise_helper, param_list):
                pbar.update()


# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Main script to create stimuli for training and testing the CNN'
    )
    parser.add_argument(
        '--directory', '-d',
        type=str,
        default='tmp',
        help='directory, where to put the data (default: %(default)s)'
    )
    parser.add_argument(
        '--length', '-l',
        type=int,
        default=1024,
        help='length of noise input (default: %(default)s)'
    )
    parser.add_argument(
        '--num_noise', '-n',
        type=int,
        default=10,
        help='number of noise files per ir file (default: %(default)s)'
    )
    parser.add_argument(
        '--jobs', '-j',
        type=int,
        default=mp.cpu_count(),
        help='Number of parallel jobs used for spatialising (default: %(default)s)'
    )
    parser.add_argument(
        '--verbose', '-v',
        default=False,
        action='store_true',
        help='Verbose mode (default: %(default)s)'
    )

    args = parser.parse_args()

    main(args)
