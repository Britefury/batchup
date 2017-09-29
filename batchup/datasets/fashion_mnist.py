import os
from .. import config
from . import mnist, dataset

_FASHION_MNIST_BASE_URL = \
    'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
_H5_FILENAME = 'fashion-mnist.h5'


_SHA256_TRAIN_IMAGES = \
    '3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84'
_SHA256_TRAIN_LABELS = \
    'a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845'
_SHA256_TEST_IMAGES = \
    '346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073'
_SHA256_TEST_LABELS = \
    '67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5'

_TRAIN_X_SRC = dataset.DownloadSourceFile(
    'train-images-idx3-ubyte.gz', base_url=_FASHION_MNIST_BASE_URL,
    sha256=_SHA256_TRAIN_IMAGES)
_TRAIN_Y_SRC = dataset.DownloadSourceFile(
    'train-labels-idx1-ubyte.gz', base_url=_FASHION_MNIST_BASE_URL,
    sha256=_SHA256_TRAIN_LABELS)
_TEST_X_SRC = dataset.DownloadSourceFile(
    't10k-images-idx3-ubyte.gz', base_url=_FASHION_MNIST_BASE_URL,
    sha256=_SHA256_TEST_IMAGES)
_TEST_Y_SRC = dataset.DownloadSourceFile(
    't10k-labels-idx1-ubyte.gz', base_url=_FASHION_MNIST_BASE_URL,
    sha256=_SHA256_TEST_LABELS)

_SOURCES = [_TRAIN_X_SRC, _TRAIN_Y_SRC, _TEST_X_SRC, _TEST_Y_SRC]


@dataset.fetch_and_convert_dataset(_SOURCES, _H5_FILENAME)
def fetch_fashion_mnist(source_paths, target_path):
    return mnist._convert_mnist('Fashion MNIST', target_path, *source_paths)


def delete_cache():  # pragma: no cover
    dataset.delete_dataset_cache(_H5_FILENAME)


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
        h5_path = fetch_fashion_mnist()
        if h5_path is not None:
            super(FashionMNIST, self).__init__(h5_path, n_val, val_lower,
                                               val_upper)
        else:
            raise RuntimeError('Could not load Fashion MNIST dataset')
