import os
import sys
import shutil
import tarfile
import pickle
import numpy as np
import tables

from .. import config
from ..image.utils import ImageArrayUInt8ToFloat32
from . import dataset


# Pickle encoding parameters depend on Python version
_PICKLE_ENC = {} if sys.version_info[0] == 2 else {'encoding': 'latin1'}

_CIFAR10_BASE_URL = 'http://www.cs.toronto.edu/~kriz/'
_SHA256_CIFAR10_TARBALL = \
    '6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce'
_H5_FILENAME = 'cifar10.h5'


_CIFAR10_SRC = dataset.DownloadSourceFile(
    'cifar-10-python.tar.gz', base_url=_CIFAR10_BASE_URL,
    sha256=_SHA256_CIFAR10_TARBALL)


@dataset.fetch_and_convert_dataset([_CIFAR10_SRC], _H5_FILENAME)
def fetch_cifar10(source_paths, target_path):
    tarball_path = source_paths[0]

    # Get the paths to the member files
    download_dir = os.path.dirname(tarball_path)
    data_paths = [os.path.join(download_dir,
                               'cifar-10-batches-py',
                               'data_batch_{}'.format(i))
                  for i in range(1, 6)]
    test_path = os.path.join(download_dir,
                             'cifar-10-batches-py', 'test_batch')

    # Unpack
    print('Unpacking CIFAR-10 tarball {}'.format(tarball_path))
    tarfile.open(name=tarball_path, mode='r:gz').extractall(
        path=download_dir)

    # Create HDF5 output file
    f_out = tables.open_file(target_path, mode='w')
    g_out = f_out.create_group(f_out.root, 'cifar10', 'MNIST data')

    print('Converting CIFAR-10 training set to HDF5')
    train_X_u8 = []
    train_y = []
    for batch_path in data_paths:
        print('Converting {} to HDF5'.format(batch_path))
        batch = pickle.load(open(batch_path, 'rb'), **_PICKLE_ENC)
        train_X_u8.append(batch['data'].reshape((-1, 3, 32, 32)))
        train_y.append(np.array(batch['labels'], dtype=np.int32))
    train_X_u8 = np.concatenate(train_X_u8, axis=0)
    train_y = np.concatenate(train_y, axis=0)
    f_out.create_array(g_out, 'train_X_u8', train_X_u8)
    f_out.create_array(g_out, 'train_y', train_y)

    print('Converting CIFAR-10 test set to HDF5')
    tst_batch = pickle.load(open(test_path, 'rb'), **_PICKLE_ENC)
    test_X_u8 = tst_batch['data'].reshape((-1, 3, 32, 32))
    test_y = np.array(tst_batch['labels'], dtype=np.int32)
    f_out.create_array(g_out, 'test_X_u8', test_X_u8)
    f_out.create_array(g_out, 'test_y', test_y)

    f_out.close()

    # Remove the contents unpacked from the tarball
    shutil.rmtree(os.path.join(download_dir, 'cifar-10-batches-py'))

    return target_path


def delete_cache():  # pragma: no cover
    dataset.delete_dataset_cache(_H5_FILENAME)


class CIFAR10 (object):
    def __init__(self, n_val=10000, val_lower=0.0, val_upper=1.0):
        h5_path = fetch_cifar10()
        if h5_path is not None:
            f = tables.open_file(h5_path, mode='r')

            train_X_u8 = f.root.cifar10.train_X_u8
            train_y = f.root.cifar10.train_y
            self.test_X_u8 = f.root.cifar10.test_X_u8
            self.test_y = f.root.cifar10.test_y

            if n_val == 0 or n_val is None:
                self.train_X_u8 = train_X_u8
                self.train_y = train_y
                self.val_X_u8 = np.zeros((0, 3, 32, 32), dtype=np.uint8)
                self.val_y = np.zeros((0,), dtype=np.int32)
            else:
                self.train_X_u8 = train_X_u8[:-n_val]
                self.val_X_u8 = train_X_u8[-n_val:]
                self.train_y, self.val_y = train_y[:-n_val], train_y[-n_val:]
        else:
            raise RuntimeError('Could not load CIFAR-10 dataset')

        self.train_X = ImageArrayUInt8ToFloat32(self.train_X_u8, val_lower,
                                                val_upper)
        self.val_X = ImageArrayUInt8ToFloat32(self.val_X_u8, val_lower,
                                              val_upper)
        self.test_X = ImageArrayUInt8ToFloat32(self.test_X_u8, val_lower,
                                               val_upper)
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                            'dog', 'frog', 'horse', 'ship', 'truck']
