
from . import mnist

_FASHION_MNIST_BASE_URL = \
    'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
_H5_FILENAME_FASHION = 'fashion-mnist.h5'


_SHA256_TRAIN_IMAGES = \
    '08dc20ab1689590a0bcd66e874647b6ee8464e2d501b5a3f1f78831db19a3fdc'
_SHA256_TRAIN_LABELS = \
    'b0197879cbda89f3dc7b894f9fd52b858e68ea4182b6947c9d8c2b67e5f18dcc'
_SHA256_TEST_IMAGES = \
    '78dcbbfb5a27efaf1b6c1d616caca68560be8766c5bffcb2791df9273a534229'
_SHA256_TEST_LABELS = \
    '42bd18137a62d5998cdeae52bf3a0676ac6b706f5cf8439b47bb5b151ae3dccf'


class FashionMNIST (mnist.MNISTBase):
    """
    Fashion-MNIST dataset

    https://github.com/zalandoresearch/fashion-mnist

    https://arxiv.org/abs/1708.07747

    @online{xiao2017/online,
      author       = {Han Xiao and Kashif Rasul and Roland Vollgraf},
      title        = {Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms},
      date         = {2017-08-28},
      year         = {2017},
      eprintclass  = {cs.LG},
      eprinttype   = {arXiv},
      eprint       = {cs.LG/1708.07747},
    }
    """
    def __init__(self, n_val=10000, val_lower=0.0, val_upper=1.0):
        h5_path = mnist._load_mnist(
            _FASHION_MNIST_BASE_URL, _H5_FILENAME_FASHION, 'Fashion MNIST',
            _SHA256_TRAIN_IMAGES, _SHA256_TRAIN_LABELS,
            _SHA256_TEST_IMAGES, _SHA256_TEST_LABELS
        )
        if h5_path is not None:
            super(FashionMNIST, self).__init__(h5_path, n_val, val_lower,
                                               val_upper)
        else:
            raise RuntimeError('Could not load Fashion MNIST dataset')
