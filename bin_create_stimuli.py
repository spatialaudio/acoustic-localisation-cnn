import argparse
import itertools
import multiprocessing
import os
import tools
import tqdm
import urllib.request


def spatialise_signal_helper(ir_stim):
    ir_file, stim_file = ir_stim

    ir_file_wo_ending = ir_file.rsplit('.', 1)[0]
    stim_file_wo_ending = stim_file.rsplit('.', 1)[0]
    stim_file_wo_dir = stim_file_wo_ending.rsplit('/', 1)[-1]

    out_file = ir_file_wo_ending + '_' + stim_file_wo_dir + '.sofa'

    tools.audio.spatialise_signal(
        ir_file,
        out_file,
        noise_length=2**10,
        sum_directions=False
    )


def sync_remote_file(url, path):
    # Split on the rightmost / and take everything on the right side of that
    name = url.rsplit('/', 1)[-1]

    # Combine the name and the downloads directory to get the local filename
    filename = os.path.join(path, name)

    if not os.path.isfile(filename):
        urllib.request.urlretrieve(url, filename)

    return filename


def process_directory(path, ir_urls, stim_urls=None, num_cores=1):

    # create directory, if it does not exist
    if not os.path.exists(path):
        os.makedirs(path)

    ir_filenames = [sync_remote_file(ir, path) for ir in ir_urls]

    if stim_urls is None:
        num_noise_in = 10
        stim_filenames = [
            os.path.join(path, 'noise{0}.wav'.format(odx))
            for odx in range(num_noise_in)
        ]
        
    else:
        stim_filenames = [sync_remote_file(stim, path) for stim in stim_urls]

    iter_allcombs = list(
        itertools.product(ir_filenames, stim_filenames)
    )

    with multiprocessing.Pool(num_cores) as p:
        with tqdm.tqdm(total=len(iter_allcombs)) as pbar:
            for _ in p.imap_unordered(spatialise_signal_helper, iter_allcombs):
                pbar.update()


def main(args):

    train_path = os.path.join(args.directory, "train")
    test_path = os.path.join(args.directory, "test")

    train_ir_remotes = [
        'https://zenodo.org/record/55418/files/QU_KEMAR_anechoic_2m.sofa'
    ]

    test_ir_remotes = [
        'https://zenodo.org/record/160749/files/QU_KEMAR_Auditorium3.sofa'
    ]

    process_directory(train_path, train_ir_remotes, num_cores=args.jobs)
    process_directory(test_path, test_ir_remotes, num_cores=args.jobs)


# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Main script for running training of CNN'
    )
    parser.add_argument(
        '--directory', '-d',
        type=str,
        default='tmp/',
        help='directory, where to put the data (default: %(default)s)'
    )
    parser.add_argument(
        '--jobs', '-j',
        type=int,
        default=multiprocessing.cpu_count(),
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
