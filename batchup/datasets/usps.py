__author__ = 'Britefury'

import gzip
import numpy as np


from britefury_lasagne import dataset


def _download_usps(filename, source='http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/'):
    return dataset.download_data(filename, source + filename)


def _load_usps_file(path):
    # Each file is a GZIP compressed text file, each line of which consists of:
    # ground truth class (as a float for some reason) followed by 256 values that are the pixel values
    # in the range [-1, 1]
    X = []
    y = []
    # Open file via gzip
    with gzip.open(path, 'r') as f:
        for line in f.readlines():
            sample = line.strip().split()
            y.append(int(float(sample[0])))
            X.append(np.array([float(val) for val in sample[1:]], dtype=np.float32).reshape((1, 1, 16, 16)))
    y = np.array(y).astype(np.int32)
    X = np.concatenate(X, axis=0).astype(np.float32)
    # Scale from [-1, 1] range to [0, 1]
    return X * 0.5 + 0.5, y


def _load_usps():
    # Download if necessary
    train_path = _download_usps('zip.train.gz')
    test_path = _download_usps('zip.test.gz')

    train_X, train_y = _load_usps_file(train_path)
    test_X, test_y = _load_usps_file(test_path)

    return train_X, train_y, test_X, test_y


class USPS (object):
    def __init__(self, n_val=729):
        train_X, train_y, test_X, test_y = _load_usps()
        if n_val == 0 or n_val is None:
            self.train_X, self.train_y = train_X, train_y
            self.val_X = np.zeros((0, 1, 16, 16), dtype=np.float32)
            self.val_y = np.zeros((0,), dtype=np.int32)
        else:
            self.train_X, self.val_X = train_X[:-n_val], train_X[-n_val:]
            self.train_y, self.val_y = train_y[:-n_val], train_y[-n_val:]
        self.test_X, self.test_y = test_X, test_y
