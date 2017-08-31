# Code adapted from:
# https://raw.githubusercontent.com/Lasagne/Lasagne/master/examples/mnist.py
import os
import numpy as np
import gzip
import tables

from .. import config
from ..image.utils import ImageArrayUInt8ToFloat32


_MNIST_DIGITS_BASE_URL = 'http://yann.lecun.com/exdb/mnist/'

_SHA256_TRAIN_IMAGES = \
    '440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609'
_SHA256_TRAIN_LABELS = \
    '3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c'
_SHA256_TEST_IMAGES = \
    '8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6'
_SHA256_TEST_LABELS = \
    'f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6'
_H5_FILENAME_DIGITS = 'mnist.h5'


def _download_mnist(filename, sha256, source):
    temp_filename = os.path.join('temp', filename)
    return config.download_data(temp_filename, source + filename, sha256)


def _read_mnist_images(path):
    # Read the inputs in Yann LeCun's binary format.
    with gzip.open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    # The inputs are vectors now, we reshape them to monochrome 2D images,
    # following the shape convention: (examples, channels, rows, columns)
    return data.reshape(-1, 1, 28, 28)


def _read_mnist_labels(path):
    # Read the labels in Yann LeCun's binary format.
    with gzip.open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    # The labels are vectors of integers now, that's exactly what we want.
    return data.astype(np.int32)


def _load_mnist(base_url, h5_filename, dataset_name, train_X_sha256,
                train_y_sha256, test_X_sha256, test_y_sha256):
    h5_path = config.get_data_path(h5_filename)
    if not os.path.exists(h5_path):
        # Download MNIST binary files
        train_X_path = _download_mnist('train-images-idx3-ubyte.gz',
                                       train_X_sha256, source=base_url)
        train_y_path = _download_mnist('train-labels-idx1-ubyte.gz',
                                       train_y_sha256, source=base_url)
        test_X_path = _download_mnist('t10k-images-idx3-ubyte.gz',
                                      test_X_sha256, source=base_url)
        test_y_path = _download_mnist('t10k-labels-idx1-ubyte.gz',
                                      test_y_sha256, source=base_url)

        if train_X_path is not None and train_y_path is not None and \
                test_X_path is not None and test_y_path is not None:
            print('Convering {} to HDF5'.format(dataset_name))
            train_X_u8 = _read_mnist_images(train_X_path)
            train_y = _read_mnist_labels(train_y_path)
            test_X_u8 = _read_mnist_images(test_X_path)
            test_y = _read_mnist_labels(test_y_path)

            f_out = tables.open_file(h5_path, mode='w')
            g_out = f_out.create_group(f_out.root, 'mnist', 'MNIST data')
            f_out.create_array(g_out, 'train_X_u8', train_X_u8)
            f_out.create_array(g_out, 'train_y', train_y)
            f_out.create_array(g_out, 'test_X_u8', test_X_u8)
            f_out.create_array(g_out, 'test_y', test_y)

            f_out.close()

            os.remove(train_X_path)
            os.remove(train_y_path)
            os.remove(test_X_path)
            os.remove(test_y_path)
        else:
            return None

    return h5_path


class MNISTBase (object):
    def __init__(self, h5_path, n_val=10000, val_lower=0.0, val_upper=1.0):
        f = tables.open_file(h5_path, mode='r')

        train_X_u8 = f.root.mnist.train_X_u8
        train_y = f.root.mnist.train_y
        self.test_X_u8 = f.root.mnist.test_X_u8
        self.test_y = f.root.mnist.test_y

        if n_val == 0 or n_val is None:
            self.train_X_u8 = train_X_u8
            self.train_y = train_y
            self.val_X_u8 = np.zeros((0, 1, 28, 28), dtype=np.uint8)
            self.val_y = np.zeros((0,), dtype=np.int32)
        else:
            self.train_X_u8 = train_X_u8[:-n_val]
            self.val_X_u8 = train_X_u8[-n_val:]
            self.train_y, self.val_y = train_y[:-n_val], train_y[-n_val:]

        self.train_X = ImageArrayUInt8ToFloat32(self.train_X_u8, val_lower,
                                                val_upper)
        self.val_X = ImageArrayUInt8ToFloat32(self.val_X_u8, val_lower,
                                              val_upper)
        self.test_X = ImageArrayUInt8ToFloat32(self.test_X_u8, val_lower,
                                               val_upper)


class MNIST (MNISTBase):
    def __init__(self, n_val=10000, val_lower=0.0, val_upper=1.0):
        h5_path = _load_mnist(
            _MNIST_DIGITS_BASE_URL, _H5_FILENAME_DIGITS, 'MNIST',
            _SHA256_TRAIN_IMAGES, _SHA256_TRAIN_LABELS,
            _SHA256_TEST_IMAGES, _SHA256_TEST_LABELS
        )
        if h5_path is not None:
            super(MNIST, self).__init__(h5_path, n_val, val_lower, val_upper)
        else:
            raise RuntimeError('Could not load MNIST dataset')
