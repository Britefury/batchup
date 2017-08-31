import pytest
import numpy as np


def test_val_0():
    from batchup.datasets import cifar10

    ds = cifar10.CIFAR10(n_val=0)

    assert ds.train_X.shape == (50000, 3, 32, 32)
    assert ds.train_X.dtype == np.float32

    assert ds.train_y.shape == (50000,)
    assert ds.train_y.dtype == np.int32

    assert ds.val_X.shape == (0, 3, 32, 32)
    assert ds.val_X.dtype == np.float32

    assert ds.val_y.shape == (0,)
    assert ds.val_y.dtype == np.int32

    assert ds.test_X.shape == (10000, 3, 32, 32)
    assert ds.test_X.dtype == np.float32

    assert ds.test_y.shape == (10000,)
    assert ds.test_y.dtype == np.int32

    assert ds.class_names == ['airplane', 'automobile', 'bird', 'cat',
                              'deer', 'dog', 'frog', 'horse', 'ship',
                              'truck']


def test_val_10k():
    from batchup.datasets import cifar10

    ds = cifar10.CIFAR10(n_val=10000)

    assert ds.train_X.shape == (40000, 3, 32, 32)
    assert ds.train_X.dtype == np.float32

    assert ds.train_y.shape == (40000,)
    assert ds.train_y.dtype == np.int32

    assert ds.val_X.shape == (10000, 3, 32, 32)
    assert ds.val_X.dtype == np.float32

    assert ds.val_y.shape == (10000,)
    assert ds.val_y.dtype == np.int32

    assert ds.test_X.shape == (10000, 3, 32, 32)
    assert ds.test_X.dtype == np.float32

    assert ds.test_y.shape == (10000,)
    assert ds.test_y.dtype == np.int32

    assert ds.class_names == ['airplane', 'automobile', 'bird', 'cat',
                              'deer', 'dog', 'frog', 'horse', 'ship',
                              'truck']


def test_train_test_split():
    from batchup.datasets import cifar10
    from batchup.tests.dataset_test_helpers import sample_hashes

    ds = cifar10.CIFAR10(n_val=0)

    train_h = sample_hashes(ds.train_X_u8)
    test_h = sample_hashes(ds.test_X_u8)

    assert set(train_h).intersection(set(test_h)) == set()
