import pytest
import numpy as np


def test_val_0():
    from batchup.datasets import usps

    ds = usps.USPS(n_val=0)

    assert ds.train_X.shape == (7291, 1, 16, 16)
    assert ds.train_X.dtype == np.float32

    assert ds.train_y.shape == (7291,)
    assert ds.train_y.dtype == np.int32

    assert ds.val_X.shape == (0, 1, 16, 16)
    assert ds.val_X.dtype == np.float32

    assert ds.val_y.shape == (0,)
    assert ds.val_y.dtype == np.int32

    assert ds.test_X.shape == (2007, 1, 16, 16)
    assert ds.test_X.dtype == np.float32

    assert ds.test_y.shape == (2007,)
    assert ds.test_y.dtype == np.int32


def test_val_729():
    from batchup.datasets import usps

    ds = usps.USPS(n_val=729)

    assert ds.train_X.shape == (6562, 1, 16, 16)
    assert ds.train_X.dtype == np.float32

    assert ds.train_y.shape == (6562,)
    assert ds.train_y.dtype == np.int32

    assert ds.val_X.shape == (729, 1, 16, 16)
    assert ds.val_X.dtype == np.float32

    assert ds.val_y.shape == (729,)
    assert ds.val_y.dtype == np.int32

    assert ds.test_X.shape == (2007, 1, 16, 16)
    assert ds.test_X.dtype == np.float32

    assert ds.test_y.shape == (2007,)
    assert ds.test_y.dtype == np.int32
