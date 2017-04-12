import numpy as np
from scipy.io import loadmat

from . import dataset


def _download_svhn(filename,
                   source='http://ufldl.stanford.edu/housenumbers/'):
    return dataset.download_data(filename, source + filename)


def _load_svhn(filename):
    # Download if necessary
    path = _download_svhn(filename)

    # Load in the Matlab file
    data = loadmat(path)

    X = data['X'].astype(np.float32) / np.float32(255.0)
    X = X.transpose(3, 2, 0, 1)
    y = data['y'].astype(np.int32)[:, 0]
    y[y == 10] = 0
    return X, y


class SVHN (object):
    def __init__(self, n_val=10000):
        train_X, train_y = _load_svhn('train_32x32.mat')
        if n_val == 0 or n_val is None:
            self.train_X, self.train_y = train_X, train_y
            self.val_X = np.zeros((0, 3, 32, 32), dtype=np.float32)
            self.val_y = np.zeros((0,), dtype=np.int32)
        else:
            self.train_X, self.val_X = train_X[:-n_val], train_X[-n_val:]
            self.train_y, self.val_y = train_y[:-n_val], train_y[-n_val:]
        self.test_X, self.test_y = _load_svhn('test_32x32.mat')


class SVHNExtra (object):
    def __init__(self):
        self.extra_X, self.extra_y = _load_svhn('extra_32x32.mat')
