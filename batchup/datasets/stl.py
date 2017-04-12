import os
import tarfile
import numpy as np

from . import dataset


def _download_stl(filename='stl10_binary.tar.gz',
                  source='http://ai.stanford.edu/~acoates/stl10/'):
    return dataset.download_data(filename, source + filename)


def _convert_batch(b):
    return (b['data'].reshape((-1, 3, 32, 32)),
            np.array(b['labels'], dtype=np.int32))


def _load_stl(filename='stl10_binary.tar.gz', load_unlabeled=False):
    # Download if necessary
    path = _download_stl(filename)

    train_X_path = os.path.join(dataset.get_dataset_dir(),
                                'stl10_binary', 'train_X.bin')
    train_y_path = os.path.join(dataset.get_dataset_dir(),
                                'stl10_binary', 'train_y.bin')
    test_X_path = os.path.join(dataset.get_dataset_dir(),
                               'stl10_binary', 'test_X.bin')
    test_y_path = os.path.join(dataset.get_dataset_dir(),
                               'stl10_binary', 'test_y.bin')
    unlabeled_X_path = os.path.join(dataset.get_dataset_dir(),
                                    'stl10_binary', 'unlabeled_X.bin')
    class_names_path = os.path.join(dataset.get_dataset_dir(),
                                    'stl10_binary', 'class_names.txt')
    fold_indices_path = os.path.join(dataset.get_dataset_dir(),
                                     'stl10_binary', 'fold_indices.txt')
    paths = [train_X_path, train_y_path, test_X_path, test_y_path,
             unlabeled_X_path, class_names_path, fold_indices_path]

    # Determine if they have been unpacked
    unpacked = True
    for p in paths:
        if not os.path.exists(p):
            print('Did not find {}'.format(p))
            unpacked = False
            break

    # Unpack if they are not there
    if not unpacked:
        print('unpacking')
        tarfile.open(name=path, mode='r:gz').extractall(
            path=dataset.get_dataset_dir())

    # Load them
    train_X = np.fromfile(open(train_X_path, 'rb'), dtype=np.uint8)
    train_X = train_X.reshape((-1, 3, 96, 96)).transpose(0, 1, 3, 2)
    train_y = np.fromfile(open(train_y_path, 'rb'), dtype=np.uint8)
    train_y = train_y.astype(np.int32) - 1
    test_X = np.fromfile(open(test_X_path, 'rb'), dtype=np.uint8)
    test_X = test_X.reshape((-1, 3, 96, 96)).transpose(0, 1, 3, 2)
    test_y = np.fromfile(open(test_y_path, 'rb'), dtype=np.uint8)
    test_y = test_y.astype(np.int32) - 1
    if load_unlabeled:
        unlabeled_X = np.fromfile(open(unlabeled_X_path, 'rb'),
                                  dtype=np.uint8)
        unlabeled_X = unlabeled_X.reshape((-1, 3, 96, 96))
        unlabeled_X = unlabeled_X.transpose(0, 1, 3, 2)
    else:
        unlabeled_X = None

    class_names = [n.strip()
                   for n in open(class_names_path, 'r').readlines()]

    train_fold_indices = []
    for fold_line in open(fold_indices_path, 'r').readlines():
        fold_ndx = np.array([int(x)
                             for x in fold_line.strip().split()])
        train_fold_indices.append(fold_ndx)

    return (train_X, train_y, test_X, test_y,
            unlabeled_X, class_names, train_fold_indices)


class STL (object):
    def __init__(self, n_val_folds=1):
        (tr_X, tr_y, tst_X, tst_y, unl_X, cls_names, tr_folds) = _load_stl()
        if n_val_folds == 0 or n_val_folds is None:
            self.train_X_u8, self.train_y = tr_X, tr_y
            self.val_X_u8 = np.zeros((0, 3, 96, 96), dtype=np.uint8)
            self.val_y = np.zeros((0,), dtype=np.int32)
        else:
            print('sizes={}'.format([f.shape for f in tr_folds]))
            val_indices = np.concatenate(tr_folds[-n_val_folds:], axis=0)
            val_mask = np.zeros(tr_y.shape, dtype=bool)
            val_mask[val_indices] = True
            self.train_X_u8 = tr_X[~val_mask]
            self.val_X_u8 = tr_X[val_mask]
            self.train_y = tr_y[~val_mask]
            self.val_y = tr_y[val_mask]
        self.test_X_u8, self.test_y = tst_X, tst_y
        self.class_names = cls_names
