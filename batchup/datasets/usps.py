import numpy as np
import tables

from .. import config


_USPS_SOURCE = 'https://github.com/Britefury/usps_dataset/raw/master/usps.h5'
_USPS_HASH = None


def _download_usps(source=_USPS_SOURCE):
    return config.download_data('usps.h5', source, _USPS_HASH)


def _load_usps():
    # Download if necessary
    data_path = _download_usps()

    f = tables.open_file(data_path, mode='r')

    train_X = f.root.usps.train_X
    train_y = f.root.usps.train_y
    test_X = f.root.usps.test_X
    test_y = f.root.usps.test_y

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
