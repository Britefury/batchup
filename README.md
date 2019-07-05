# BatchUp
Python library for extracting mini-batches of data from a data source for the purpose of training neural networks.

Quick example:

```py3
from batchup import data_source

# Construct an array data source
ds = data_source.ArrayDataSource([train_X, train_y])

# Iterate over samples, drawing batches of 64 elements in random order
for (batch_X, batch_y) in ds.batch_iterator(batch_size=64, shuffle=True):
    # Processes batches here...
```

Documentation available at https://batchup.readthedocs.io


# Table of Contents

#### Installation

#### Batch iteration
Processing data in mini-batches:
- quick batch iteration; a basic example
- iterating over subsets identified by indices
- data augmentation
- including sample indices in the mini-batches
- infinite batch iteration; an iterator that generates batches endlessly
- sample weighting to alter likelihood of samples (e.g. to compensate for class imbalance)
- iterating over two data sets simultaneously where their sizes differ (e.g. for semi-supervised learning)
- iterating over data sets that are NOT stored as NumPy arrays (e.g. on disk or generated on the fly)
- parallel processing to speed up iteration where loading/preparing samples could be slow

#### Gathering results and loss values
- removing the for-loop; predict values for samples in one line
- computing mean loss/error values

#### Standard datasets
BatchUp supports some standard machine learning datasets. They will be automatically downloaded if necessary.
- MNIST
- SVHN
- CIFAR-10
- CIFAR-100
- STL
- USPS

#### Configuring BatchUp
Data paths, etc.

More details further down, but briefly, use either the `~/.batchup.cfg` configuration file or the `BATCHUP_HOME` environment varible.


## Installation

You can install BatchUp with:

```> pip install batchup```

## Batch iteration
### Quick batch iteration

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

### Iterating over subsets identified by indices

We can specify the indices of a subset of the samples in a dataset and draw mini-batches from only those samples:

```py3
import numpy as np

# Randomly choose a subset of 20,000 samples, by indices
subset_a = np.random.permutation(train_X.shape[0])[:20000]

# Construct an array data source that will only draw samples whose indices are in `subset_a`
ds = data_source.ArrayDataSource([train_X, train_y], indices=subset_a)

# Drawing batches of 64 elements in random order
for (batch_X, batch_y) in ds.batch_iterator(batch_size=64, shuffle=np.random.RandomState(12345)):
    # Processes batches here...
```

### Data augmentation

We can define a function that applies data augmentation on the fly. Let's assume that `train_X` contains image data,
has the shape `(sample, channel, height, width)` and that we wish to horizontally flip some of the images:

```py3
import numpy as np

# Define our batch augmentation function.
def augment_batch(batch_X, batch_y):
    # Create an array that selects samples with 50% probability and convert to `bool` dtype
    flip_flags = np.random.binomial(1, 0.5, size=(len(batch_X),)) != 0
    
    # Flip the width dimension in selected samples
    batch_X[flip_flags, ...] = flip_flags[flip_flags, :, :, ::-1]
    
    # Return the batch as a tuple
    return batch_X, batch_y
    
# Construct an array data source that will only draw samples whose indices are in `subset_a`
ds = data_source.ArrayDataSource([train_X, train_y])

# Apply augmentation
ds = ds.map(augment_batch)

# Drawing batches of 64 elements in random order
for (batch_X, batch_y) in ds.batch_iterator(batch_size=64, shuffle=np.random.RandomState(12345)):
    # Processes batches here...
```

More complex augmentation may incurr significant runtime cost. This can be alleviated by preparing batches
in background threads. See the *parallel processing* section below.

### Including sample indices in the mini-batches

We can ask to be provided with the indices of the samples that were drawn to form the mini-batch:

```py3
# Construct an array data source that will provide sample indices
ds = data_source.ArrayDataSource([train_X, train_y], include_indices=True)

# Drawing batches of 64 elements in random order
for (batch_ndx, batch_X, batch_y) in ds.batch_iterator(batch_size=64, shuffle=np.random.RandomState(12345)):
    # Processes batches here...
```

### Infinite batch iteration

Lets say you need an iterator that extracts samples from your dataset and starts from the beginning when it reaches the end:

```py3
ds = data_source.ArrayDataSource([train_X, train_y], repeats=-1)
```

Now use the `batch_iterator` method as before.

The `repeats` parameter accepts either `-1` for infininte, or any positive integer `>= 1` for a specified number of repetitions.

This will also work if the dataset has less samples than the batch size; not a common use case but can happen in certain situations involving semi-supervised learning for instance.

### Sample weighting to alter likelihood of samples

If you want some samples to be drawn more frequently than others, construct a `sampling.WeightedSampler` and pass
it as the `sampler` argument to the `ArrayDataSource` constructor. In the example the per-sample weights are stored
in `train_w`.

```py3
from batchup import sampling

sampler = sampling.WeightedSampler(weights=train_w)

ds = data_source.ArrayDataSource([train_X, train_y], sampler=sampler)

# Drawing batches of 64 elements in random order
for (batch_X, batch_y) in ds.batch_iterator(batch_size=64, shuffle=np.random.RandomState(12345)):
    # Processes batches here...
```

**Note** that in-order is NOT supported when using `sampling.WeightedSampler`, so `shuffle` *cannot* be `False` or
`None`.

To draw from a subset of the dataset, use `sampling.WeightedSubsetSampler`:

```py3
from batchup import sampling

# NOTE that that parameter is called `sub_weights` (rather than `weights`) and that it must have the
# same length as `indices`.
sampler = sampling.WeightedSubsetSampler(sub_weights=train_w[subset_a], indices=subset_a)

ds = data_source.ArrayDataSource([train_X, train_y], sampler=sampler)

# Drawing batches of 64 elements in random order
for (batch_X, batch_y) in ds.batch_iterator(batch_size=64, shuffle=np.random.RandomState(12345)):
    # Processes batches here...
```


#### Class balancing helper

An alternate constructor `sampling.WeightedSampler.class_balancing_sampler` is available to construct a weighted sampler to compensate for class imbalance:

```py3
# Construct the sampler; NOTE that the `n_classes` argument is *optional*
sampler = sampling.WeightedSampler.class_balancing_sampler(y=train_y, n_classes=train_y.max() + 1)

ds = data_source.ArrayDataSource([train_X, train_y], sampler=sampler)

# Drawing batches of 64 elements in random order
for (batch_X, batch_y) in ds.batch_iterator(batch_size=64, shuffle=np.random.RandomState(12345)):
    # Processes batches here...
```

The `sampling.WeightedSampler.class_balancing_sample_weights` helper method constructs an array of sample weights,
in case you wish to modify the weights first:
```py3
weights = sampling.WeightedSampler.class_balancing_sample_weights(y=train_y, n_classes=train_y.max() + 1)

# Assume `modify_weights` is defined above
weights = modify_weights(weights)

# Construct the sampler and the data source
sampler = sampling.WeightedSampler(weights=weights)
ds = data_source.ArrayDataSource([train_X, train_y], sampler=sampler)

# Drawing batches of 64 elements in random order
for (batch_X, batch_y) in ds.batch_iterator(batch_size=64, shuffle=np.random.RandomState(12345)):
    # Processes batches here...
```


### Iterating over two data sources of wildly different sizes for semi-supervised learning

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

### Using data that is NOT stored as NumPy arrays

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

### Using parallel processing to speed things up

The above example has a potential performance problem as loading the images from disk would introduce latency. We can use the `work_pool` module to prepare the mini-batches in separate threads or processes to hide this latency.

#### Using threads

The modifications to the previous example to use parallel processing are quite simple (lets assume that the `LoadImagesFromDisk` class is defined and that `train_X`, `train_y` and `ds` (an `ArrayDataSource` instance) have already been built:

```py3
from batchup import work_pool

# Build a pool of 4 worker threads:
th_pool = work_pool.WorkerThreadPool(processes=4)

# Construct a data source that prepares mini-batches in the background
# It wraps the existing data source `ds` and will try to keep a buffer of 32
# mini-batches full to eliminate latency:
par_ds = th_pool.parallel_data_source(ds, batch_buffer_size=32)

# As soon as we create an iterator, it will start filling its buffer; lets create an
# iterator right now to get it going in the background:
par_iter = par_ds.batch_iterator(batch_size=64, shuffle=np.random.RandomState(12345))

# Do some other initialisation stuff that may take a while...

# By now, with any luck, some batches will be ready to retrieve
                
for (batch_X, batch_y) in par_iter:
    # Process batches here...
```

#### Using processes

In some cases the data source that you wish to parallelize may include some cacheing logic that is not thread safe. In such cases you can use process based pools that use separate processes rather than threads.
There are one or two gotchas, namely that using process-based pools entails a higher overhead and that the data source class and its dependent types must be declared in the top level of a module so that `pickle` can find them.

```py3
# Build a pool of 4 worker processes:
proc_pool = work_pool.WorkerProcessPool(processes=4)

# Construct a data source that prepares mini-batches in the background
# It wraps the existing non-thread-safe data source `ds` and
# will try to keep a buffer of 32 mini-batches full to eliminate latency:
par_ds = proc_pool.parallel_data_source(ds, batch_buffer_size=32)

# ... use `par_ds` the same way as before ...
```

## Gathering results and loss values

We can further simplify training and evaluation procedures using the `batch_map_concat` and `batch_map_mean` methods.

### Removing the for-loop; predict values for samples in one line

Lets assume we have a prediction function `f_pred` of the form `f_pred(batch_X) -> batch_pred_y`. 
If we want to predict results for our test set in `test_X`, we can do this in one line, without the for loop:

```py3
test_ds = data_source.ArrayDataSource([test_X])

(pred_y,) = test_ds.batch_map_concat(f_pred, batch_size=256)
```

The `batch_map_concat` method will process all the samples in `test_X` and gather the results in a tuple of arrays, hence
the `(pred_y,) = ...`. If you want `tqdm` ([PyPi](http://pypi.python.org/pypi/tqdm), [GitHub](http://github.com/noamraph/tqdm)) to give you a progress bar:

```py3
(pred_y,) = test_ds.batch_map_concat(f_pred, batch_size=256, progress_iter_func=tqdm.tqdm)
```

### Computing mean loss/error values

Lets assume we have a evaluation function `f_eval` of the form `f_eval(batch_X, batch_y) -> [log_loss_sum, err_count]`. 
Assuming that we are doing classification, `f_eval` returns the sum of the per-sample log-losses and the number of errors.
The `batch_map_mean` method will process all of the data in the data source, gather loss and error counts and return the mean:

```py3
val_ds = data_source.ArrayDataSource([val_X, val_y])

mean_log_loss, mean_err_rate = val_ds.batch_map_mean(f_eval, batch_size=256)
```

Note that as above, the `progress_iter_func` parameter can be passed `tqdm.tqdm` to give you a progress bar.


## Standard datasets

BatchUp provides support for using some standard datasets.

#### MNIST dataset

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


#### SVHN dataset

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


#### CIFAR-10 dataset

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


#### CIFAR-100 dataset

Load the CIFAR-100 dataset:
```py3
from batchup.datasets import cifar100

# Load CIFAR-100 dataset (downloading it if necessary) and retain the last 5000
# training samples for validation
ds = cifar100.CIFAR100(n_val=5000)
```

- `ds.train_X` is a `(n, 3, 32, 32)` `float32` array that contains the
    training images.
- `ds.train_y` is a `(n,)` `int32` array that contains the fine ground
    truth classes.
- `ds.train_y_coarse` is a `(n,)` `int32` array that contains the coarse
    ground truth classes.
- `ds.val_X`, `ds.val_y` and `ds.val_y_coarse` contain the validation samples
- `ds.test_X`, `ds.test_y` and `ds.test_y_coarse` contain the test samples
- `ds.class_names` lists the class names of the corresponding fine ground
    truth indices
- `ds.class_names_coarse` lists the class names of the corresponding coarse
    ground truth indices


#### STL dataset

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
    
We keep the image data in `uint8` form to save memory,


#### USPS dataset

Load the USPS dataset (similar to MNIST hand-written digits but smaller):
```py3
from batchup.datasets import usps

# Load USPS dataset (downloading it if necessary) and retain 729
# training samples for validation
ds = usps.USPS(n_val=729)
```

- `ds.train_X` is a `(n, 1, 16, 16)` `float32` array that contains the
    training images.
- `ds.train_y` is a `(n,)` `int32` array that contains the ground truths.
- `ds.val_X` and `ds.val_y` contain the validation samples
- `ds.test_X` and `ds.test_y` contain the test samples


## Configuring BatchUp (paths etc).

The configuration for BatchUp lives in `.batchup.cfg` in your home directory.

By default BatchUp will store its data (e.g. downloaded datasets) in a directory called `.batchup` that resides in your home directory. If you wish it to locate this data somewhere else (some of the datasets an take a few gigabytes), create the configuration file mentioned above:


```cfg
[paths]
data_dir=/some/path/batchup_data
```

Alternatively you can set the `BATCHUP_HOME` environment variable top the BatchUp data directory.
