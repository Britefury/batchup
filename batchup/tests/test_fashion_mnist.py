import pytest
import numpy as np


def test_val_0():
    from batchup.datasets import fashion_mnist

    ds = fashion_mnist.FashionMNIST(n_val=0)

    assert ds.train_X.shape == (60000, 1, 28, 28)
    assert ds.train_X.dtype == np.float32

    assert ds.train_y.shape == (60000,)
    assert ds.train_y.dtype == np.int32

    assert ds.val_X.shape == (0, 1, 28, 28)
    assert ds.val_X.dtype == np.float32

    assert ds.val_y.shape == (0,)
    assert ds.val_y.dtype == np.int32

    assert ds.test_X.shape == (10000, 1, 28, 28)
    assert ds.test_X.dtype == np.float32

    assert ds.test_y.shape == (10000,)
    assert ds.test_y.dtype == np.int32


def test_val_10k():
    from batchup.datasets import fashion_mnist

    ds = fashion_mnist.FashionMNIST(n_val=10000)

    assert ds.train_X.shape == (50000, 1, 28, 28)
    assert ds.train_X.dtype == np.float32

    assert ds.train_y.shape == (50000,)
    assert ds.train_y.dtype == np.int32

    assert ds.val_X.shape == (10000, 1, 28, 28)
    assert ds.val_X.dtype == np.float32

    assert ds.val_y.shape == (10000,)
    assert ds.val_y.dtype == np.int32

    assert ds.test_X.shape == (10000, 1, 28, 28)
    assert ds.test_X.dtype == np.float32

    assert ds.test_y.shape == (10000,)
    assert ds.test_y.dtype == np.int32


def test_train_test_split():
    from batchup.datasets import fashion_mnist
    from batchup.tests.dataset_test_helpers import sample_hashes

    ds = fashion_mnist.FashionMNIST(n_val=0)

    train_h = sample_hashes(ds.train_X)
    test_h = sample_hashes(ds.test_X)

    assert set(train_h).intersection(set(test_h)) == set()


def test_duplication_and_overlap():
    from batchup.datasets import fashion_mnist
    from batchup.tests.dataset_test_helpers import sample_hashes

    ds = fashion_mnist.FashionMNIST(n_val=0)

    train_h = sample_hashes(ds.train_X_u8)
    test_h = sample_hashes(ds.test_X_u8)

    train_dict = {}
    test_dict = {}
    for i, h in enumerate(train_h):
        train_dict.setdefault(h, []).append(i)
    for i, h in enumerate(test_h):
        test_dict.setdefault(h, []).append(i)

    print('Training set duplicates:')
    for k, v in train_dict.items():
        if len(v) > 1:
            for j in range(1, len(v)):
                assert (ds.train_X_u8[v[0]] == ds.train_X_u8[v[j]]).all()
            print(v)

    print('Test set duplicates:')
    for k, v in test_dict.items():
        if len(v) > 1:
            for j in range(1, len(v)):
                assert (ds.test_X_u8[v[0]] == ds.test_X_u8[v[j]]).all()
            print(v)

    print('Training set samples overlapping with test set:')
    for k, v in train_dict.items():
        if k in test_dict:
            assert (ds.train_X_u8[v[0]] == ds.test_X_u8[test_dict[k][0]]).all()
            print('Train samples {} overlap with test samples {}'.format(
                v, test_dict[k]))

    overlap = set(train_h).intersection(set(test_h))
    print(len(overlap))
    assert overlap == set()
