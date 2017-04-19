# BatchUp
Python library for extracting mini-batches of data from a data source for the purpose of training neural networks.


# Examples

We will demonstrate the following examples

- quick batch iteration; a basic example
- infinite batch iteration; an iterator that generates batches endlessly
- iterating over two data sets simultaneously where their sizes differ (e.g. for semi-supervised learning)
- iterating over data sets that are not stored as NumPy arrays (e.g. on disk or generated on the fly)
- parallel processing to speed up iteration where loading/preparing samples could be slow

## Example 1: quick batch iteration

Assume we have a training set loaded in the variables `train_X` and `train_y`:

```py3
from batchup import data_source

# Construct an array data source
ds = data_source.ArrayDataSource([train_X, train_y])

# Iterate over samples, drawing batches of 64 elements in random order
for (batch_X, batch_y) in ds.batch_iterator(batch_size=64, shuffle=np.random.RandomState(12345)):
    # Processes batches here...
```

Some notes:
- the last batch will be short (have less samples than the requested batch size) if there isn't enough data to fill it.
- using `shuffle=True` will use NumPy's default random number generator
- not specifying shuffle will process the samples in-order

## Example 2: infinite batch iteration

Lets say you need an iterator that extracts samples from your dataset and starts from the beginning when it reaches the end:

```py3
ds = data_source.ArrayDataSource([train_X, train_y], repeats=-1)
```

Now use the `batch_iterator` method as before.

The `repeats` parameter accepts either `-1` for infininte, or any positive integer `>= 1` for a specified number of repetitions.

This will also work if the dataset has less samples than the batch size; not a common use case but can happen in certain situations involving semi-supervised learning for instance.

## Example 3: iterating over two data sources of wildly different sizes for semi-supervised learning

In semi-supervised learning we have a small dataset of labeled samples `lab_X` with ground truths `lab_y` and a larger set of unlabeled samples `unlab_X`. Lets say we want a single epoch to consist of the entire unlabeled dataset while looping over the labeled dataset repeatly. The `CompositeDataSource` class can help us here.

Without using `CompositeDataSource`:

```py3
rng = np.random.RandomState(12345)

# Construct the data sources; the labeled data source will repeat infinitely
ds_lab = data_source.ArrayDataSource([lab_X, lab_y], repeats=-1)
ds_unlab = data_source.ArrayDataSource([unlab_X])

# Construct an iterator to get samples from our labeled data source:
lab_iter = ds_lab.batch_iterator(batch_size=64, shuffle=rng)

# Iterate over the unlabled data set in the for-loop
for (batch_unlab_X,) in ds_unlab.batch_iterator(batch_size=64, shuffle=rng):
    # Extract batches from the labeled iterator ourselves
    batch_lab_X, batch_lab_y = next(lab_iter)
    
    # Process batches here...
```

Now using `CompositeDataSource`:

```py3
# Construct the data sources; the labeled data source will repeat infinitely
ds_lab = data_source.ArrayDataSource([lab_X, lab_y], repeats=-1)
ds_unlab = data_source.ArrayDataSource([unlab_X])
ds = data_source.CompositeDataSource([ds_lab, ds_unlab])

# Iterate over both the labeled and unlabeled samples:
for (batch_lab_X, batch_lab_y, batch_unlab_X) in ds.batch_iterator(batch_size=64, shuffle=rng):
    # Process batches here...

```

The two component data sources (`ds_lab` and `ds_unlab`) will be shuffled independently.

You can also have `CompositeDataSource` generate structured mini-batches that reflect the structure of the data source:

```py3
# Flatten this time round:
ds_struct = data_source.CompositeDataSource([ds_lab, ds_unlab], flatten=False)

# Iterate over both the labeled and unlabeled samples:
for ((batch_lab_X, batch_lab_y), (batch_unlab_X,)) in ds_struct.batch_iterator(batch_size=64, shuffle=rng):
    # Process batches here...
```

`CompositeDataSource` instances can be arbitrarily nested.

## Example 4: using data that is not stored as NumPy arrays

The arrays passed to `ArrayDataSource` do not have to be NumPy arrays, they just have to be array-like. An array-like object should implement the `__len__` method that returns the number of samples and the `__getitem__` method that returns the samples themselves. Note that `__getitem__` should accept integer indices, slices, or NumPy integer arrays that give the indices of the samples to retrieve.

Lets day we want to implement a data source that loads images from disk on the fly. Lets also assume that the prefix of the filename, either `'cat'` or `'dog'` gives the ground truth:

```py3
import glob
import os
from scipy.misc import imread
from batchup import data_source

class LoadImagesFromDisk (object):
    def __init__(self, paths):
        # Paths is a list of file paths
        self.paths = paths

    # We have to imlement the `__len__` method:
    def __len__(self):
        return len(self.paths)
        
        
    # We have to implement the `__getitem__` method that `ArrayDataSource` will use to get samples
    def __getitem__(self, index):
        if isinstance(index, (int, long)):
            # A single integer index; return that sample
            return imread(self.paths[index])
        elif isinstance(index, slice):
            # A slice
            images = [imread(p) for p in self.paths[index]]
            return np.concatenate([img[None, ...] for img in images], axis=0)
        elif isinstance(index, np.ndarray):
            if index.ndim != 1:
                raise ValueError('index array should only have 1 dimension, not {}'.format(index.ndim))
            images = [imread(self.paths[i]) for i in index]
            return np.concatenate([img[None, ...] for img in images], axis=0)
        else:
            raise TypeError('index should be an integer, a slice or a NumPy array, '
                            'not a {}'.format(type(index))

# Get our image paths
image_paths = glob.glob('/path/to/my/images/*.jpg')

# Build our array-like data source
train_X = LoadImagesFromDisk(image_paths)

# Construct our ground truths as a NumPy array
train_y = [(1 if os.path.basename(p).startswith('dog') else 0) for p in image_paths)]
train_y = np.array(train_y, dtype=np.int32)

# Mixing custom array types with NumPy arrays is fine
ds = data_source.ArrayDataSource([train_X, train_y])
                
for (batch_X, batch_y) in ds.batch_iterator(batch_size=64, shuffle=np.random.RandomState(12345)):
    # Process batches here...
```

## Example 5: using paralle processing to speed things up

The above example has a potential performance problem as loading the images from disk would introduce latency. We can use the `work_pool` module to prepare the mini-batches in separate processes to hide this latency.

The modifications to the previous example to use parallel processing are quite simple (lets assume that the `LoadImagesFromDisk` class is defined and that `train_X`, `train_y` and `ds` (an `ArrayDataSource` instance) have already been built:

```py3
from batchup import work_pool

# Build a pool of 4 worker processes:
pool = work_pool.WorkerPool(processes=4)

# Construct a data source that prepares mini-batches in the background
# It wraps the existing data source `ds` and will try to keep a buffer of 32
# mini-batches full to eliminate latency:
par_ds = pool.parallel_data_source(ds, batch_buffer_size=32)

# As soon as we create an iterator, it will start filling its buffer; lets create an
# iterator right now to get it going in the background:
par_iter = par_ds..batch_iterator(batch_size=64, shuffle=np.random.RandomState(12345))

# Do some other initialisation stuff that may take a while...

# By now, with any luck, some batches will be ready to retrieve
                
for (batch_X, batch_y) in par_iter:
    # Process batches here...
```

# Standard datasets

BatchUp provides support for using some standard datasets.

## MNIST dataset

Load the MNIST dataset:
```py3
from batchup.datasets import mnist

# Load MNIST dataset (downloading it if necessary) and retain the last 10000
# training samples for validation
ds = mnist.MNIST(n_val=10000)
```

- `ds.train_X` is a `(n, 1, 28, 28)` `float32` array that contains the
    training images.
- `ds.train_y` is a `(n,)` `int32` array that contains the ground truths.
- `ds.val_X` and `ds.val_y` contain the validation samples
- `ds.test_X` and `ds.test_y` contain the test samples


## SVHN dataset

Load the SVHN dataset:
```py3
from batchup.datasets import svhn

# Load SVHN dataset (downloading it if necessary) and retain the last 10000
# training samples for validation
ds = svhn.SVHN(n_val=10000)
```

- `ds.train_X` is a `(n, 3, 32, 32)` `float32` array that contains the
    training images.
- `ds.train_y` is a `(n,)` `int32` array that contains the ground truths.
- `ds.val_X` and `ds.val_y` contain the validation samples
- `ds.test_X` and `ds.test_y` contain the test samples


## CIFAR-10 dataset

Load the CIFAR-10 dataset:
```py3
from batchup.datasets import cifar10

# Load CIFAR-10 dataset (downloading it if necessary) and retain the last 5000
# training samples for validation
ds = cifar10.CIFAR10(n_val=5000)
```

- `ds.train_X` is a `(n, 3, 32, 32)` `float32` array that contains the
    training images.
- `ds.train_y` is a `(n,)` `int32` array that contains the ground truths.
- `ds.val_X` and `ds.val_y` contain the validation samples
- `ds.test_X` and `ds.test_y` contain the test samples
- `ds.class_names` lists the class names of the corresponding ground truth
    indices


## STL dataset

Load the STL dataset:
```py3
from batchup.datasets import stl

# Load STL dataset (downloading it if necessary) and retain 1 fold of
# training samples for validation
ds = stl.STL(n_val_folds=1)
```

- `ds.train_X_u8` is a `(n, 3, 96, 96)` `uint8` array that contains the
    training images.
- `ds.train_y` is a `(n,)` `int32` array that contains the ground truths.
- `ds.val_X_u8` and `ds.val_y` contain the validation samples
- `ds.test_X_u8` and `ds.test_y` contain the test samples
- `ds.class_names` lists the class names of the corresponding ground truth
    indices

