import os as _os
import urllib.request as _urlreq
from . import layers as _layers
from . import losses as _losses
import keras as _keras
import glob as _glob

def sync_remote_file(url, path):
    # Split on the rightmost / and take everything on the right side of that
    name = url.rsplit('/', 1)[-1]

    # Combine the name and the downloads directory to get the local filename
    filename = _os.path.join(path, name)

    if not _os.path.isfile(filename):
        _urlreq.urlretrieve(url, filename)

    return filename


def sync_remote_filelist(urls, path):

    # create directory, if it does not exist
    if not _os.path.exists(path):
        _os.makedirs(path)

    # sync impulse responses
    filelist = [sync_remote_file(u, path) for u in urls]

    return filelist

def load_newest_model(pattern):

    # possible custom loss functions and layers
    custom_obj_dict = {
        'SplitLayer': _layers.SplitLayer,
        'Corr1D': _layers.Corr1D,
        'jensen_shannon_divergence': _losses.jensen_shannon_divergence
    }

    latest_file = max(
        _glob.glob(pattern),  # all files matching pattern
        key=_os.path.getctime  # time of last modification
    )
    # Load model:
    return _keras.models.load_model(
        latest_file,
        custom_objects=custom_obj_dict
    )