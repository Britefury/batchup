import pytest
import numpy as np


@pytest.mark.bigdataset
def test_val_0():
    from batchup.datasets import stl

    ds = stl.STL(n_val_folds=0)

    assert ds.train_X_u8.shape == (5000, 3, 96, 96)
    assert ds.train_X_u8.dtype == np.uint8

    assert ds.train_y.shape == (5000,)
    assert ds.train_y.dtype == np.int32

    assert ds.val_X_u8.shape == (0, 3, 96, 96)
    assert ds.val_X_u8.dtype == np.uint8

    assert ds.val_y.shape == (0,)
    assert ds.val_y.dtype == np.int32

    assert ds.test_X_u8.shape == (8000, 3, 96, 96)
    assert ds.test_X_u8.dtype == np.uint8

    assert ds.test_y.shape == (8000,)
    assert ds.test_y.dtype == np.int32

    assert ds.class_names == ['airplane', 'bird', 'car', 'cat',
                              'deer', 'dog', 'horse', 'monkey', 'ship',
                              'truck']


@pytest.mark.bigdataset
def test_val_1fold():
    from batchup.datasets import stl

    ds = stl.STL(n_val_folds=1)

    assert ds.train_X_u8.shape == (4000, 3, 96, 96)
    assert ds.train_X_u8.dtype == np.uint8

    assert ds.train_y.shape == (4000,)
    assert ds.train_y.dtype == np.int32

    assert ds.val_X_u8.shape == (1000, 3, 96, 96)
    assert ds.val_X_u8.dtype == np.uint8

    assert ds.val_y.shape == (1000,)
    assert ds.val_y.dtype == np.int32

    assert ds.test_X_u8.shape == (8000, 3, 96, 96)
    assert ds.test_X_u8.dtype == np.uint8

    assert ds.test_y.shape == (8000,)
    assert ds.test_y.dtype == np.int32

    assert ds.class_names == ['airplane', 'bird', 'car', 'cat',
                              'deer', 'dog', 'horse', 'monkey', 'ship',
                              'truck']
