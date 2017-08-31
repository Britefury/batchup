import os
import numpy as np
import tables

from .. import config


_USPS_SOURCE = 'https://github.com/Britefury/usps_dataset/raw/master/usps.h5'
_USPS_HASH = 'ba768d9a9b11e79b31c1e40130647c4fc04e6afc1fb41a0d4b9f1176065482b4'
_H5_FILENAME = 'usps.h5'


def _download_usps(source=_USPS_SOURCE):
    return config.download_data(_H5_FILENAME, source, _USPS_HASH)


def delete_cache():  # pragma: no cover
    h5_path = config.get_data_path(_H5_FILENAME)
    if os.path.exists(h5_path):
        os.remove(h5_path)


class USPS (object):
    def __init__(self, n_val=729):
        data_path = _download_usps()

        if data_path is not None:
            f = tables.open_file(data_path, mode='r')

            train_X = f.root.usps.train_X
            train_y = f.root.usps.train_y
            test_X = f.root.usps.test_X
            test_y = f.root.usps.test_y

            if n_val == 0 or n_val is None:
                self.train_X, self.train_y = train_X, train_y
                self.val_X = np.zeros((0, 1, 16, 16), dtype=np.float32)
                self.val_y = np.zeros((0,), dtype=np.int32)
            else:
                self.train_X, self.val_X = train_X[:-n_val], train_X[-n_val:]
                self.train_y, self.val_y = train_y[:-n_val], train_y[-n_val:]
            self.test_X, self.test_y = test_X, test_y
