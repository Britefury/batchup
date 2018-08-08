import os
import shutil
import tarfile
import numpy as np
import tables

from .. import config
from ..image.utils import ImageArrayUInt8ToFloat32
from . import dataset


_STL_BASE_URL = 'http://ai.stanford.edu/~acoates/stl10/'
_SHA256_STL_TARBALL = \
    'f31fd99273a1acb8609c8db427cebb1de3f71de77758cdc0e22956e1289b9866'
_H5_FILENAME = 'stl10.h5'


_STL_SRC = dataset.DownloadSourceFile(
    'stl10_binary.tar.gz', base_url=_STL_BASE_URL,
    sha256=_SHA256_STL_TARBALL)


@dataset.fetch_and_convert_dataset([_STL_SRC], _H5_FILENAME)
def fetch_stl(source_paths, target_path):
    tarball_path = source_paths[0]

    download_dir = os.path.dirname(tarball_path)

    # Get the paths to the member files
    train_X_pth = os.path.join(download_dir,
                               'stl10_binary', 'train_X.bin')
    train_y_pth = os.path.join(download_dir,
                               'stl10_binary', 'train_y.bin')
    test_X_pth = os.path.join(download_dir,
                              'stl10_binary', 'test_X.bin')
    test_y_pth = os.path.join(download_dir,
                              'stl10_binary', 'test_y.bin')
    unlabeled_X_pth = os.path.join(download_dir,
                                   'stl10_binary', 'unlabeled_X.bin')
    class_names_pth = os.path.join(download_dir,
                                   'stl10_binary', 'class_names.txt')
    fold_indices_pth = os.path.join(download_dir,
                                    'stl10_binary',
                                    'fold_indices.txt')

    # Unpack
    print('Unpacking STL tarball {}'.format(tarball_path))
    tarfile.open(name=tarball_path, mode='r:gz').extractall(
        path=download_dir)

    # Create HDF5 output file
    f_out = tables.open_file(target_path, mode='w')
    g_out = f_out.create_group(f_out.root, 'stl', 'MNIST data')
    filters = tables.Filters(complevel=9, complib='blosc')

    print('Converting STL training set to HDF5')
    train_X_u8 = np.fromfile(open(train_X_pth, 'rb'), dtype=np.uint8)
    train_X_u8 = train_X_u8.reshape((-1, 3, 96, 96))
    train_X_u8 = train_X_u8.transpose(0, 1, 3, 2)
    f_out.create_array(g_out, 'train_X_u8', train_X_u8)

    train_y = np.fromfile(open(train_y_pth, 'rb'), dtype=np.uint8)
    train_y = train_y.astype(np.int32) - 1
    f_out.create_array(g_out, 'train_y', train_y)

    print('Converting STL test set to HDF5')
    test_X_u8 = np.fromfile(open(test_X_pth, 'rb'), dtype=np.uint8)
    test_X_u8 = test_X_u8.reshape((-1, 3, 96, 96))
    test_X_u8 = test_X_u8.transpose(0, 1, 3, 2)
    f_out.create_array(g_out, 'test_X_u8', test_X_u8)

    test_y = np.fromfile(open(test_y_pth, 'rb'), dtype=np.uint8)
    test_y = test_y.astype(np.int32) - 1
    f_out.create_array(g_out, 'test_y', test_y)

    print('Converting STL unlabeled set to HDF5')
    unl_X_u8 = np.fromfile(open(unlabeled_X_pth, 'rb'),
                           dtype=np.uint8)
    unl_X_u8 = unl_X_u8.reshape((-1, 3, 96, 96))
    unl_X_u8 = unl_X_u8.transpose(0, 1, 3, 2)
    unl_X_u8_arr = f_out.create_earray(
        g_out, 'unl_X_u8', tables.UInt8Atom(), (0, 3, 96, 96),
        filters=filters)
    unl_X_u8_arr.append(unl_X_u8)

    print('Converting STL class names to HDF5')
    class_names = [n.strip().encode('us-ascii')
                   for n in open(class_names_pth, 'r').readlines()]
    f_out.create_array(g_out, 'class_names', class_names)

    print('Converting STL fold indices to HDF5')
    fold_ndx_arr = f_out.create_vlarray(g_out, 'fold_indices',
                                        tables.Int32Atom())
    for fold_line in open(fold_indices_pth, 'r').readlines():
        fold_ndx = np.array([int(x)
                             for x in fold_line.strip().split()])
        fold_ndx_arr.append(fold_ndx.astype(np.int32))

    f_out.close()

    # Clean up files unpacked from tarball
    shutil.rmtree(os.path.join(download_dir, 'stl10_binary'))

    return target_path


def delete_cache():  # pragma: no cover
    dataset.delete_dataset_cache(_H5_FILENAME)


class STL (object):
    def __init__(self, n_val_folds=1, val_lower=0.0, val_upper=1.0):
        h5_path = fetch_stl()
        if h5_path is not None:
            f = tables.open_file(h5_path, mode='r')

            train_X_u8 = f.root.stl.train_X_u8
            train_y = f.root.stl.train_y
            self.test_X_u8 = f.root.stl.test_X_u8
            self.test_y = f.root.stl.test_y
            self.unl_X_u8 = f.root.stl.unl_X_u8
            tr_folds = f.root.stl.fold_indices
            tr_folds = [tr_folds[i] for i in range(len(tr_folds))]
            self.tr_folds = tr_folds

            if n_val_folds == 0 or n_val_folds is None:
                self.train_X_u8 = train_X_u8
                self.train_y = train_y
                self.val_X_u8 = np.zeros((0, 3, 96, 96), dtype=np.uint8)
                self.val_y = np.zeros((0,), dtype=np.int32)
            else:
                val_indices = np.concatenate(tr_folds[-n_val_folds:], axis=0)
                val_mask = np.zeros((len(train_y),), dtype=bool)
                val_mask[val_indices] = True
                train_indices = np.arange(len(train_y))[~val_mask]
                self.train_X_u8 = train_X_u8[train_indices, :, :, :]
                self.val_X_u8 = train_X_u8[val_indices, :, :, :]
                self.train_y = train_y[train_indices]
                self.val_y = train_y[val_indices]

            self.class_names = list(f.root.stl.class_names)

        else:
            raise RuntimeError('Could not load MNIST dataset')

        self.train_X = ImageArrayUInt8ToFloat32(self.train_X_u8, val_lower,
                                                val_upper)
        self.val_X = ImageArrayUInt8ToFloat32(self.val_X_u8, val_lower,
                                              val_upper)
        self.test_X = ImageArrayUInt8ToFloat32(self.test_X_u8, val_lower,
                                               val_upper)
