"""
Image utils module
"""
import numpy as np


class ImageArrayUInt8ToFloat32 (object):
    """
    An array wrapper class that converts `uint8` image values in the range
    0:255 to `float32` with the range being (optionally) provided to the
    constructor. The default range is 0:1.

    Note: read-only
    """
    def __init__(self, arr_u8, val_lower=0.0, val_upper=1.0):
        """
        Constructor

        Parameters
        ----------
        arr_8: array-like object e.g. a NumPy array
            An object that supports `__len__` and `__getitem__` and
            optionally has a shape attribute
        val_lower: float
            The lower end of the output value range
        val_upper: float
            The upper end of the output value range
        """
        self.arr_u8 = arr_u8
        self.offset = np.float32(val_lower)
        self.scale = np.float32((val_upper - val_lower) / 255.0)
        self.dtype = np.float32

    @property
    def shape(self):
        """
        Shape attribute accessor

        Returns
        -------
        The shape of the underlying array if available, otherwise
        raises `AttributeError`
        """
        try:
            return self.arr_u8.shape
        except AttributeError:
            raise AttributeError('Underlying array has no `shape` attribute')

    def __len__(self):
        """
        Length accessor

        Returns
        -------
        The length of the underlying array.
        """
        return len(self.arr_u8)

    def __getitem__(self, *indices):
        """
        Item accessor

        Accesses elements from the underlying array and converts them to
        `float32` type.

        Parameters
        ----------
        indices
            Indices into the underlying array

        Returns
        -------
        Items from underlying array converted to `float32`
        """
        x = self.arr_u8.__getitem__(*indices)
        return x.astype(np.float32) * self.scale + self.offset
