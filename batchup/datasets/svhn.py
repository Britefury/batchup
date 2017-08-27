import os
import numpy as np
from scipy.io import loadmat
import tables

from .. import config


def _download_svhn(filename, sha256,
                   source='http://ufldl.stanford.edu/housenumbers/'):
    temp_filename = os.path.join('temp', filename)
    return config.download_data(temp_filename, source + filename, sha256)


_H5_TRAIN_TEST_FILENAME = 'svhn_train_test.h5'
_H5_EXTRA_FILENAME = 'svhn_extra.h5'
_SHA256_TRAIN = \
    b'435e94d69a87fde4fd4d7f3dd208dfc32cb6ae8af2240d066de1df7508d083b8'
_SHA256_TEST = \
    b'cdce80dfb2a2c4c6160906d0bd7c68ec5a99d7ca4831afa54f09182025b6a75b'
_SHA256_EXTRA = \
    None


def _svhn_matlab_to_h5(h5_X, h5_y, m_X, m_y):
    m_X = m_X.transpose(3, 2, 0, 1)
    for i in range(0, len(m_y), 10240):
        j = min(i + 10240, len(m_y))
        batch_y = m_y[i:j].astype(np.int32)[:, 0]
        batch_y[batch_y == 10] = 0
        h5_X.append(m_X[i:j].astype(np.float32) / np.float32(255.0))
        h5_y.append(batch_y)


def _load_svhn_train_test():
    h5_path = config.get_data_path(_H5_TRAIN_TEST_FILENAME)
    if not os.path.exists(h5_path):
        # Download SVHN Matlab files
        train_path = _download_svhn('train_32x32.mat', _SHA256_TRAIN)
        test_path = _download_svhn('test_32x32.mat', _SHA256_TEST)

        if train_path is not None and test_path is not None:
            f_out = tables.open_file(h5_path, mode='w')
            g_out = f_out.create_group(f_out.root, 'svhn', 'SVHN data')
            filters = tables.Filters(complevel=9, complib='blosc')
            train_X_arr = f_out.create_earray(g_out, 'train_X', tables.Float32Atom(), (0, 3, 32, 32), filters=filters)
            train_y_arr = f_out.create_earray(g_out, 'train_y', tables.Int32Atom(), (0,), filters=filters)
            test_X_arr = f_out.create_earray(g_out, 'test_X', tables.Float32Atom(), (0, 3, 32, 32), filters=filters)
            test_y_arr = f_out.create_earray(g_out, 'test_y', tables.Int32Atom(), (0,), filters=filters)

            # Load in the training data Matlab file
            print('Converting {} to HDF5...'.format(train_path))
            data = loadmat(train_path)
            _svhn_matlab_to_h5(train_X_arr, train_y_arr, data['X'], data['y'])
            del data

            # Load in the test data Matlab file
            print('Converting {} to HDF5...'.format(test_path))
            data = loadmat(test_path)
            _svhn_matlab_to_h5(test_X_arr, test_y_arr, data['X'], data['y'])
            del data

            f_out.close()

            os.remove(train_path)
            os.remove(test_path)
        else:
            return None

    return h5_path


def _load_svhn_extra():
    h5_path = config.get_data_path(_H5_EXTRA_FILENAME)
    if not os.path.exists(h5_path):
        # Download SVHN Matlab file
        extra_path = _download_svhn('extra_32x32.mat', _SHA256_EXTRA)

        if extra_path is not None:
            print('Converting {} to HDF5...'.format(extra_path))
            f_out = tables.open_file(h5_path, mode='w')
            g_out = f_out.create_group(f_out.root, 'svhn', 'SVHN data')
            filters = tables.Filters(complevel=9, complib='blosc')
            X_arr = f_out.create_earray(g_out, 'extra_X', tables.Float32Atom(), (0, 3, 32, 32), filters=filters)
            y_arr = f_out.create_earray(g_out, 'extra_y', tables.Int32Atom(), (0,), filters=filters)

            # Load in the extra data Matlab file
            data = loadmat(extra_path)
            _svhn_matlab_to_h5(X_arr, y_arr, data['X'], data['y'])
            del data

            f_out.close()

            os.remove(extra_path)
        else:
            return None

    return h5_path


class SVHN (object):
    def __init__(self, n_val=10000):
        h5_path = _load_svhn_train_test()
        if h5_path is not None:
            f = tables.open_file(h5_path, mode='r')

            train_X = f.root.svhn.train_X
            train_y = f.root.svhn.train_y
            self.test_X = f.root.svhn.test_X
            self.test_y = f.root.svhn.test_y

            if n_val == 0 or n_val is None:
                self.train_X, self.train_y = train_X, train_y
                self.val_X = np.zeros((0, 3, 32, 32), dtype=np.float32)
                self.val_y = np.zeros((0,), dtype=np.int32)
            else:
                self.train_X, self.val_X = train_X[:-n_val], train_X[-n_val:]
                self.train_y, self.val_y = train_y[:-n_val], train_y[-n_val:]
        else:
            raise RuntimeError('Could not load SVHN dataset')


class SVHNExtra (object):
    def __init__(self):
        h5_path = _load_svhn_extra()
        if h5_path is not None:
            f = tables.open_file(h5_path, mode='r')

            self.X = f.root.svhn.X
            self.y = f.root.svhn.y
        else:
            raise RuntimeError('Could not load SVHN dataset')
