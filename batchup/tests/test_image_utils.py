import pytest
import numpy as np


def test_ImageArrayUInt8ToFloat32():
    from batchup.image.utils import ImageArrayUInt8ToFloat32

    x = np.array([[0, 127, 255], [30, 60, 90]], dtype=np.uint8)
    xf32 = x.astype(np.float32) * np.float32(1.0 / 255.0)

    assert x.shape == (2, 3)

    x_wrapped = ImageArrayUInt8ToFloat32(x)
    x_wrapped_cen = ImageArrayUInt8ToFloat32(x, val_lower=-1.0,
                                             val_upper=1.0)

    assert x_wrapped.shape == (2, 3)
    assert len(x_wrapped) == 2

    assert np.allclose(x_wrapped[...], xf32)

    # Check that scaling the range works
    xf32_cen = x.astype(np.float32) * np.float32(2.0 / 255.0) - 1.0
    assert np.allclose(x_wrapped_cen[...], xf32_cen)

    # An array-like object without a shape parameter
    class ArrayLike (object):
        def __len__(self):
            return 10

        def __getitem__(self, index):
            return np.arange(10).astype(np.uint8)[index]

    # Check the basics
    a = ArrayLike()
    a_wrapped = ImageArrayUInt8ToFloat32(a)

    assert len(a_wrapped) == 10
    assert a_wrapped[0] == 0.0

    # Now ensure that accessing the `shape` attribute raises `AttributeError`
    with pytest.raises(AttributeError):
        shape = a_wrapped.shape
        del shape
