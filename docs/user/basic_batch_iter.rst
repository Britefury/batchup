Basic batch iteration from arrays
=================================

BatchUp defines *data sources* from which we draw data in mini-batches. They are defined in the
:py:mod:`.data_source` module and the one that you will most frequency use is :py:class:`.data_source.ArrayDataSource`.


.. contents::


Simple batch iteration from NumPy arrays
----------------------------------------

This example will show how to draw mini-batches of samples from NumPy arrays in random order.

Assume we have a training set in the form of NumPy arrays in the variables ``train_X`` and ``train_y``.

First, construct a *data source* that will draw data from ``train_X`` and ``train_y``:

.. code:: python

    from batchup import data_source

    # Construct an array data source
    ds = data_source.ArrayDataSource([train_X, train_y])

:py:class:`.ArrayDataSource` notes:
    - ``train_X`` and ``train_y`` must have the same number of samples
    - you can use any number of arrays when building the :py:class:`.ArrayDataSource`

Now we can use the :py:meth:`~.RandomAccessDataSource.batch_iterator` method to create a batch iterator
from which we can draw mini-batches of data:

.. code:: python

    # Iterate over samples, drawing batches of 64 elements in
    # random order
    for (batch_X, batch_y) in ds.batch_iterator(
            batch_size=64, shuffle=np.random.RandomState(12345)):
        # Processes batch_X and batch_y here...

Batch iterator notes:
    - the last batch will be short (have less samples than the requested batch size) if there isn't enough data to fill it
    - the ``shuffle`` parameter:
        - using ``shuffle=True`` will use NumPy's default random number generator
        - if no value is provided for ``shuffle``, samples will be processed in-order

**Note**: we don't *have* to use NumPy arrays; any array-like object will do; see :ref:`array-like-objects` for more.



Iterating over a subset of the samples
--------------------------------------

We can specify the indices of a subset of the samples in a dataset and draw mini-batches from only those samples:

.. code:: python

    import numpy as np

    # Randomly choose a subset of 20,000 samples, by indices
    subset_a = np.random.permutation(len(train_X))[:20000]

    # Construct an array data source that will only draw samples whose indices are in `subset_a`
    ds = data_source.ArrayDataSource([train_X, train_y], indices=subset_a)

    # Drawing batches of 64 elements in random order
    for (batch_X, batch_y) in ds.batch_iterator(
            batch_size=64, shuffle=np.random.RandomState(12345)):
        # Processes batches here...


Getting the indices of sample in the mini-batches
-------------------------------------------------

We can ask to be provided with the indices of the samples that were drawn to form the mini-batch:

.. code:: python

    # Construct an array data source that will provide sample indices
    ds = data_source.ArrayDataSource([train_X, train_y], include_indices=True)

    # Drawing batches of 64 elements in random order
    for (batch_ndx, batch_X, batch_y) in ds.batch_iterator(
            batch_size=64, shuffle=np.random.RandomState(12345)):
        # Processes batches here; indices in batch_ndx


Batches from repeated/looped arrays
-----------------------------------

Lets say you need an iterator that extracts samples from your dataset and starts from the beginning when it reaches
the end. Provide a value for the ``repeats`` argument of the :py:class:`.ArrayDataSource` constructor like so:

.. code:: python

    ds_times_5 = data_source.ArrayDataSource([train_X, train_y], repeats=5)

Now use the :py:meth:`~.RandomAccessDataSource.batch_iterator` method as before.

The ``repeats`` parameter accepts either ``-1`` for infinite, or any positive integer ``>= 1`` for a specified
number of repetitions:

.. code:: python

    inf_ds = data_source.ArrayDataSource([train_X, train_y], repeats=-1)

This will also work if the dataset has less samples than the batch size; this is not a common use case but it can
happen.

