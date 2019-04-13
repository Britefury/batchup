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

_CIFAR100_BASE_URL = 'http://www.cs.toronto.edu/~kriz/'
_SHA256_CIFAR100_TARBALL = \
    '85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7'
_H5_FILENAME = 'cifar100.h5'


_CIFAR100_SRC = dataset.DownloadSourceFile(
    'cifar-100-python.tar.gz', base_url=_CIFAR100_BASE_URL,
    sha256=_SHA256_CIFAR100_TARBALL)


@dataset.fetch_and_convert_dataset([_CIFAR100_SRC], _H5_FILENAME)
def fetch_cifar100(source_paths, target_path):
    tarball_path = source_paths[0]

    # Get the paths to the member files
    download_dir = os.path.dirname(tarball_path)
    train_path = os.path.join(download_dir,
                              'cifar-100-python', 'train')
    test_path = os.path.join(download_dir,
                             'cifar-100-python', 'test')
    meta_path = os.path.join(download_dir,
                             'cifar-100-python', 'meta')

    # Unpack
    print('Unpacking CIFAR-100 tarball {}'.format(tarball_path))
    tarfile.open(name=tarball_path, mode='r:gz').extractall(
        path=download_dir)

    # Create HDF5 output file
    f_out = tables.open_file(target_path, mode='w')
    g_out = f_out.create_group(f_out.root, 'cifar100', 'CIFAR-100 data')

    print('Converting CIFAR-100 training set to HDF5')
    train_batch = pickle.load(open(train_path, 'rb'), **_PICKLE_ENC)
    train_X_u8 = train_batch['data'].reshape((-1, 3, 32, 32))
    train_y = np.array(train_batch['fine_labels'], dtype=np.int32)
    train_y_coarse = np.array(train_batch['coarse_labels'], dtype=np.int32)
    f_out.create_array(g_out, 'train_X_u8', train_X_u8)
    f_out.create_array(g_out, 'train_y', train_y)
    f_out.create_array(g_out, 'train_y_coarse', train_y_coarse)

    print('Converting CIFAR-100 test set to HDF5')
    tst_batch = pickle.load(open(test_path, 'rb'), **_PICKLE_ENC)
    test_X_u8 = tst_batch['data'].reshape((-1, 3, 32, 32))
    test_y = np.array(tst_batch['fine_labels'], dtype=np.int32)
    test_y_coarse = np.array(tst_batch['coarse_labels'], dtype=np.int32)
    f_out.create_array(g_out, 'test_X_u8', test_X_u8)
    f_out.create_array(g_out, 'test_y', test_y)
    f_out.create_array(g_out, 'test_y_coarse', test_y_coarse)

    print('Converting CIFAR-100 metadata to HDF5')
    meta = pickle.load(open(meta_path, 'rb'), **_PICKLE_ENC)
    class_names = meta['fine_label_names']
    class_names_coarse = meta['coarse_label_names']
    f_out.create_array(g_out, 'class_names', class_names)
    f_out.create_array(g_out, 'class_names_coarse', class_names_coarse)

    f_out.close()

    # Remove the contents unpacked from the tarball
    shutil.rmtree(os.path.join(download_dir, 'cifar-100-python'))

    return target_path


def delete_cache():  # pragma: no cover
    dataset.delete_dataset_cache(_H5_FILENAME)


class CIFAR100 (object):
    def __init__(self, n_val=10000, val_lower=0.0, val_upper=1.0):
        h5_path = fetch_cifar100()
        if h5_path is not None:
            f = tables.open_file(h5_path, mode='r')

            train_X_u8 = f.root.cifar100.train_X_u8
            train_y = f.root.cifar100.train_y
            train_y_coarse = f.root.cifar100.train_y_coarse
            self.test_X_u8 = f.root.cifar100.test_X_u8
            self.test_y = f.root.cifar100.test_y
            self.test_y_coarse = f.root.cifar100.test_y_coarse
            self.class_names = dataset.classnames_from_h5(
                f.root.cifar100.class_names)
            self.class_names_coarse = dataset.classnames_from_h5(
                f.root.cifar100.class_names_coarse)

            if n_val == 0 or n_val is None:
                self.train_X_u8 = train_X_u8
                self.train_y = train_y
                self.train_y_coarse = train_y_coarse
                self.val_X_u8 = np.zeros((0, 3, 32, 32), dtype=np.uint8)
                self.val_y = np.zeros((0,), dtype=np.int32)
                self.val_y_coarse = np.zeros((0,), dtype=np.int32)
            else:
                self.train_X_u8 = train_X_u8[:-n_val]
                self.val_X_u8 = train_X_u8[-n_val:]
                self.train_y, self.val_y = train_y[:-n_val], train_y[-n_val:]
                self.train_y_coarse, self.val_y_coarse = \
                    (train_y_coarse[:-n_val], train_y_coarse[-n_val:])
        else:
            raise RuntimeError('Could not load CIFAR-10 dataset')

        self.train_X = ImageArrayUInt8ToFloat32(self.train_X_u8, val_lower,
                                                val_upper)
        self.val_X = ImageArrayUInt8ToFloat32(self.val_X_u8, val_lower,
                                              val_upper)
        self.test_X = ImageArrayUInt8ToFloat32(self.test_X_u8, val_lower,
                                               val_upper)
