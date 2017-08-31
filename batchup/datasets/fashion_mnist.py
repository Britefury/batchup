import os
from .. import config
from . import mnist

_FASHION_MNIST_BASE_URL = \
    'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
_H5_FILENAME = 'fashion-mnist.h5'


_SHA256_TRAIN_IMAGES = \
    None
_SHA256_TRAIN_LABELS = \
    None
_SHA256_TEST_IMAGES = \
    None
_SHA256_TEST_LABELS = \
    None


def delete_cache():  # pragma: no cover
    h5_path = config.get_data_path(_H5_FILENAME)
    if os.path.exists(h5_path):
        os.remove(h5_path)


class FashionMNIST (mnist.MNISTBase):
    """
    Fashion-MNIST dataset

    https://github.com/zalandoresearch/fashion-mnist

    https://arxiv.org/abs/1708.07747

    @online{xiao2017/online,
      author       = {Han Xiao and Kashif Rasul and Roland Vollgraf},
      title        = {Fashion-MNIST: a Novel Image Dataset for Benchmarking
                      Machine Learning Algorithms},
      date         = {2017-08-28},
      year         = {2017},
      eprintclass  = {cs.LG},
      eprinttype   = {arXiv},
      eprint       = {cs.LG/1708.07747},
    }
    """
    def __init__(self, n_val=10000, val_lower=0.0, val_upper=1.0):
        h5_path = mnist._load_mnist(
            _FASHION_MNIST_BASE_URL, _H5_FILENAME, 'Fashion MNIST',
            _SHA256_TRAIN_IMAGES, _SHA256_TRAIN_LABELS,
            _SHA256_TEST_IMAGES, _SHA256_TEST_LABELS
        )
        if h5_path is not None:
            super(FashionMNIST, self).__init__(h5_path, n_val, val_lower,
                                               val_upper)
        else:
            raise RuntimeError('Could not load Fashion MNIST dataset')
