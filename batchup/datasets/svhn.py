import os
import numpy as np
from scipy.io import loadmat
import tables

from .. import config
from ..image.utils import ImageArrayUInt8ToFloat32


def _download_svhn(filename, sha256,
                   source='http://ufldl.stanford.edu/housenumbers/'):
    temp_filename = os.path.join('temp', filename)
    return config.download_data(temp_filename, source + filename, sha256)


_SHA256_TRAIN_MAT = \
    '435e94d69a87fde4fd4d7f3dd208dfc32cb6ae8af2240d066de1df7508d083b8'
_SHA256_TEST_MAT = \
    'cdce80dfb2a2c4c6160906d0bd7c68ec5a99d7ca4831afa54f09182025b6a75b'
_SHA256_EXTRA_MAT = \
    None
_H5_TRAIN_TEST_FILENAME = 'svhn_train_test.h5'
_H5_EXTRA_FILENAME = 'svhn_extra.h5'


def _read_svhn_matlab(mat_path):
    mat = loadmat(mat_path)
    m_X = mat['X'].astype(np.uint8).transpose(3, 2, 0, 1)
    m_y = mat['y'].astype(np.int32)[:, 0]
    m_y[m_y == 10] = 0
    return m_X, m_y


def _load_svhn_train_test():
    h5_path = config.get_data_path(_H5_TRAIN_TEST_FILENAME)
    if not os.path.exists(h5_path):
        # Download SVHN Matlab files
        train_path = _download_svhn('train_32x32.mat', _SHA256_TRAIN_MAT)
        test_path = _download_svhn('test_32x32.mat', _SHA256_TEST_MAT)

        if train_path is not None and test_path is not None:
            f_out = tables.open_file(h5_path, mode='w')
            g_out = f_out.create_group(f_out.root, 'svhn', 'SVHN data')

            # Load in the training data Matlab file
            print('Converting {} to HDF5...'.format(train_path))
            train_X_u8, train_y = _read_svhn_matlab(train_path)
            f_out.create_array(g_out, 'train_X_u8', train_X_u8)
            f_out.create_array(g_out, 'train_y', train_y)
            del train_X_u8
            del train_y

            # Load in the test data Matlab file
            print('Converting {} to HDF5...'.format(test_path))
            test_X_u8, test_y = _read_svhn_matlab(test_path)
            f_out.create_array(g_out, 'test_X_u8', test_X_u8)
            f_out.create_array(g_out, 'test_y', test_y)
            del test_X_u8
            del test_y

            f_out.close()

            os.remove(train_path)
            os.remove(test_path)
        else:
            return None

    return h5_path


def _svhn_matlab_to_h5(h5_X, h5_y, mat_path):
    mat = loadmat(mat_path)
    m_X = mat['X']
    m_y = mat['y']
    m_X = m_X.transpose(3, 2, 0, 1)
    for i in range(0, len(m_y), 10240):
        j = min(i + 10240, len(m_y))
        batch_y = m_y[i:j].astype(np.int32)[:, 0]
        batch_y[batch_y == 10] = 0
        h5_X.append(m_X[i:j].astype(np.uint8))
        h5_y.append(batch_y)


def _load_svhn_extra():
    h5_path = config.get_data_path(_H5_EXTRA_FILENAME)
    if not os.path.exists(h5_path):
        # Download SVHN Matlab file
        extra_path = _download_svhn('extra_32x32.mat', _SHA256_EXTRA_MAT)

        if extra_path is not None:
            print('Converting {} to HDF5 (compressed)...'.format(extra_path))
            f_out = tables.open_file(h5_path, mode='w')
            g_out = f_out.create_group(f_out.root, 'svhn', 'SVHN data')
            filters = tables.Filters(complevel=9, complib='blosc')
            X_u8_arr = f_out.create_earray(
                g_out, 'extra_X_u8', tables.UInt8Atom(), (0, 3, 32, 32),
                filters=filters)
            y_arr = f_out.create_earray(
                g_out, 'extra_y', tables.Int32Atom(), (0,), filters=filters)

            # Load in the extra data Matlab file
            _svhn_matlab_to_h5(X_u8_arr, y_arr, extra_path)

            f_out.close()

            os.remove(extra_path)
        else:
            return None

    return h5_path


class SVHN (object):
    def __init__(self, n_val=10000, val_lower=0.0, val_upper=1.0):
        h5_path = _load_svhn_train_test()
        if h5_path is not None:
            f = tables.open_file(h5_path, mode='r')

            train_X_u8 = f.root.svhn.train_X_u8
            train_y = f.root.svhn.train_y
            self.test_X_u8 = f.root.svhn.test_X_u8
            self.test_y = f.root.svhn.test_y

            if n_val == 0 or n_val is None:
                self.train_X_u8 = train_X_u8
                self.train_y = train_y
                self.val_X_u8 = np.zeros((0, 3, 32, 32), dtype=np.uint8)
                self.val_y = np.zeros((0,), dtype=np.int32)
            else:
                self.train_X_u8 = train_X_u8[:-n_val]
                self.train_y = train_y[:-n_val]
                self.val_X_u8 = train_X_u8[-n_val:]
                self.val_y = train_y[-n_val:]
        else:
            raise RuntimeError('Could not load SVHN dataset')

        self.train_X = ImageArrayUInt8ToFloat32(self.train_X_u8, val_lower,
                                                val_upper)
        self.val_X = ImageArrayUInt8ToFloat32(self.val_X_u8, val_lower,
                                              val_upper)
        self.test_X = ImageArrayUInt8ToFloat32(self.test_X_u8, val_lower,
                                               val_upper)


class SVHNExtra (object):
    def __init__(self, val_lower=0.0, val_upper=1.0):
        h5_path = _load_svhn_extra()
        if h5_path is not None:
            f = tables.open_file(h5_path, mode='r')

            self.X_u8 = f.root.svhn.extra_X_u8
            self.y = f.root.svhn.extra_y
        else:
            raise RuntimeError('Could not load SVHN extra dataset')

        self.X = ImageArrayUInt8ToFloat32(self.X_u8, val_lower, val_upper)
