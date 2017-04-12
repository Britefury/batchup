import pytest
import numpy as np


def test_val_0():
    from batchup.datasets import svhn

    ds = svhn.SVHN(n_val=0)

    assert ds.train_X.shape == (73257, 3, 32, 32)
    assert ds.train_X.dtype == np.float32

    assert ds.train_y.shape == (73257,)
    assert ds.train_y.dtype == np.int32

    assert ds.val_X.shape == (0, 3, 32, 32)
    assert ds.val_X.dtype == np.float32

    assert ds.val_y.shape == (0,)
    assert ds.val_y.dtype == np.int32

    assert ds.test_X.shape == (26032, 3, 32, 32)
    assert ds.test_X.dtype == np.float32

    assert ds.test_y.shape == (26032,)
    assert ds.test_y.dtype == np.int32


def test_val_10k():
    from batchup.datasets import svhn

    ds = svhn.SVHN(n_val=10000)

    assert ds.train_X.shape == (63257, 3, 32, 32)
    assert ds.train_X.dtype == np.float32

    assert ds.train_y.shape == (63257,)
    assert ds.train_y.dtype == np.int32

    assert ds.val_X.shape == (10000, 3, 32, 32)
    assert ds.val_X.dtype == np.float32

    assert ds.val_y.shape == (10000,)
    assert ds.val_y.dtype == np.int32

    assert ds.test_X.shape == (26032, 3, 32, 32)
    assert ds.test_X.dtype == np.float32

    assert ds.test_y.shape == (26032,)
    assert ds.test_y.dtype == np.int32
