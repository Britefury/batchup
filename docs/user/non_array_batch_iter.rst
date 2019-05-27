Loading images from disk on-the-fly: data from array-like objects
=================================================================

So far we've only seen how to extract batches from NumPy arrays. This is not always possible or convenient. For
example we may wish to load images from files on disk on-the-fly, as loading all of the images into memory
may exhaust the available RAM. We will now demonstrate how load images on-the-fly from files on-disk.


.. contents::


.. _array-like-objects:

Data from array-like objects (data accessors)
---------------------------------------------

:py:class:`.data_source.ArrayDataSource` is quite flexible; it can draw data from array-like objects, not just
NumPy arrays. This allows us to customise how the data is acquired.

Our array-like object must define the following:

- a ``__len__`` method that returns the number of available samples
- a ``__getitem__`` method that retrieves samples identified by index; the index can either be an integer or a NumPy
  array of indices that identify multiple samples to retrieve in one go

Lets define an array-like image accessor that is given a list of paths that identify images to load. We will
define a helper method that will use Pillow to load the image:

.. code:: python

    import six
    import numpy as np
    from PIL import Image

    class ImageFileAccessor (object):
        def __init__(self, paths):
            # Paths is a list of file paths
            self.paths = paths

        # We have to imlement the `__len__` method:
        def __len__(self):
            return len(self.paths)

        # We have to implement the `__getitem__` method that
        # `ArrayDataSource` will use to get samples
        def __getitem__(self, index):
            # Check if its a built-in Python integer *or* a NumPy
            # integer type
            if isinstance(index, six.integer_types + (np.integer,)):
                return self._load_image(self.paths[index])
            elif isinstance(index, np.ndarray):
                if index.ndim != 1:
                    raise ValueError(
                        'index array should only have 1 dimension, '
                        'not {}'.format(index.ndim))
                images = [self._load_image(self.paths[i]) for i in index]
                return np.stack(images, axis=0)
            else:
                raise TypeError('index should be an integer or a NumPy '
                                'array, not a {}'.format(type(index))

        # Define a helper method for loading an image
        def _load_image(self, path):
            # Load the image using Pillow and convert to a NumPy array
            img = Image.open(path)
            return np.array(img)


We can now gather the paths of some image files and pass them to ``ImageFileAccessor`` and use it
in place of a NumPy array:

.. code:: python

    import glob
    import os

    # Put your dataset path here
    PATH = '/path/to/images'

    # Get image paths
    image_paths = glob.glob(os.path.join(PATH, '*.jpg'))

    # Build our array-like image file accessor
    train_X = ImageFileAccessor(image_paths)

    # Let's assume `train_y` is a NumPy array

    # Mixing custom array types with NumPy arrays is fine
    ds = data_source.ArrayDataSource([train_X, train_y])

    for (batch_X, batch_y) in ds.batch_iterator(
            batch_size=64, shuffle=np.random.RandomState(12345)):
        # Process batches here...


.. _array_like_objects_lists:

Lists instead of arrays
-----------------------

The above code will work fine if all the image have the same resolution. If they are of varying sizes
``np.stack`` will fail.

Also ``__getitem__`` doesn't have to return NumPy arrays; it can return a single PIL image or a list of PIL images:

.. code:: python

    import six
    import numpy as np
    from PIL import Image

    class NonUniformImageFileAccessor (object):
        def __init__(self, paths):
            # Paths is a list of file paths
            self.paths = paths

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, index):
            if isinstance(index, six.integer_types + (np.integer,)):
                return self._load_image(self.paths[index])
            elif isinstance(index, np.ndarray):
                if index.ndim != 1:
                    raise ValueError('index array should only have 1 dimension, '
                        'not {}'.format(index.ndim))
                return [self._load_image(self.paths[i])
                        for i in index]
            else:
                raise TypeError('index should be an integer or a NumPy '
                                'array, not a {}'.format(type(index))

        # Define a helper method for loading an image
        def _load_image(self, path):
            # Load the image using Pillow
            return Image.open(path)


Lets load the `Kaggle Dogs vs Cats <https://www.kaggle.com/c/dogs-vs-cats>`_ training set. Also, lets
define a batch augmentation function (see :doc:`data_augmentation`) that will scale each image to ``64 x 64``
pixels and convert it to a NumPy array:

.. code:: python

    def augment_batch(batch_X, batch_y):
        out_X = []
        # For each PIL Image in `batch_X`:
        for img in batch_X:
            # Resize to 64 x 64
            img = img.resize((64, 64))

            # PIL allows you to easily get the image data as
            # a NumPy array
            x = np.array(img)

            out_X.append(x)

        # Stack the images into one array
        out_X = np.stack(out_X, axis=0)

        return (out_X, batch_y)


    # Put your dataset path here
    PATH = '/path/to/dogs_vs_cats'

    # Get paths to the training set images
    image_paths = glob.glob(os.path.join(PATH, 'train', '*.jpg'))

    # Build our array-like image file accessor
    train_X = NonUniformImageFileAccessor(image_paths)

    # Construct our ground truths as a NumPy array
    # The ground truth class is determined by the prefix
    train_y = [(1 if os.path.basename(p).startswith('dog') else 0)
               for p in image_paths)]
    train_y = np.array(train_y, dtype=np.int32)

    # Mixing custom array types with NumPy arrays is fine
    kaggle_ds = data_source.ArrayDataSource([train_X, train_y])

    # Apply augmentation function
    kaggle_ds = kaggle_ds.map(augment_batch)

    for (batch_X, batch_y) in kaggle_ds.batch_iterator(
            batch_size=64, shuffle=np.random.RandomState(12345)):
        # Process batches here...


Performance issues
------------------

Loading images from disk in this way can incur a significant performance overhead due to disk access and
decompressing the image data once it has been loaded into memory. It would be desirable to do this in a
separate thread or process in order to hide this latency. You can learn how to do this in :doc:`parallel_batch`.
