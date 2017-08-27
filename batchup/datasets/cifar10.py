import os
import sys
import tarfile
import pickle
import numpy as np

from .. import config

PICKLE_ENC = {} if sys.version_info[0] == 2 else {'encoding': 'latin1'}


def _download_cifar10(filename='cifar-10-python.tar.gz',
                      source='http://www.cs.toronto.edu/~kriz/'):
    return config.download_data(filename, source + filename)


def _convert_batch(b):
    return (b['data'].reshape((-1, 3, 32, 32)),
            np.array(b['labels'], dtype=np.int32))


def _load_cifar10(filename='cifar-10-python.tar.gz'):
    # Download if necessary
    path = _download_cifar10(filename)

    # Get the paths to the member files
    data_paths = [os.path.join(dataset.get_dataset_dir(),
                               'cifar-10-batches-py',
                               'data_batch_{}'.format(i))
                  for i in range(1, 6)]
    test_path = os.path.join(dataset.get_dataset_dir(),
                             'cifar-10-batches-py', 'test_batch')

    # Determine if they have been unpacked
    unpacked = True
    for p in data_paths + [test_path]:
        if not os.path.exists(p):
            unpacked = False
            break

    # Unpack if they are not there
    if not unpacked:
        print('unpacking')
        tarfile.open(name=path, mode='r:gz').extractall(
            path=dataset.get_dataset_dir())

    # Load them
    data_batches = [pickle.load(open(p, 'rb'), **PICKLE_ENC)
                    for p in data_paths]
    test_batch = pickle.load(open(test_path, 'rb'), **PICKLE_ENC)

    train_X, train_y = list(zip(*[_convert_batch(b) for b in data_batches]))
    test_X, test_y = _convert_batch(test_batch)

    train_X = np.concatenate(train_X, axis=0)
    train_y = np.concatenate(train_y, axis=0)

    train_X = train_X.astype(np.float32) / np.float32(255.0)
    test_X = test_X.astype(np.float32) / np.float32(255.0)
    return train_X, train_y, test_X, test_y


class CIFAR10 (object):
    def __init__(self, n_val=10000):
        train_X, train_y, test_X, test_y = _load_cifar10()
        if n_val == 0 or n_val is None:
            self.train_X, self.train_y = train_X, train_y
            self.val_X = np.zeros((0, 3, 32, 32), dtype=np.float32)
            self.val_y = np.zeros((0,), dtype=np.int32)
        else:
            self.train_X, self.val_X = train_X[:-n_val], train_X[-n_val:]
            self.train_y, self.val_y = train_y[:-n_val], train_y[-n_val:]
        self.test_X, self.test_y = test_X, test_y
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                            'dog', 'frog', 'horse', 'ship', 'truck']
