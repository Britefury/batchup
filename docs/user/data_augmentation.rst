Data augmentation on-the-fly
============================

We can define a function that applies data augmentation on the fly. Let's assume that the images batches that we draw
have the shape ``(sample, channel, height, width)`` and that we wish to randomly choose 50% of the images in the
batch to be horizontally flipped:

.. code:: python

    # Define our batch augmentation function.
    def augment_batch(batch_X, batch_y):
        # Create an array of random 0's and 1's with 50% probability
        flip_flags = np.random.binomial(1, 0.5, size=(len(batch_X),))

        # Convert to `bool` dtype.
        flip_flags = flip_flags.astype(bool)

        # Flip the horizontal dimension in samples identified by
        # `flip_flags`
        batch_X[flip_flags, ...] = flip_flags[flip_flags, :, :, ::-1]

        # Return the batch as a tuple
        return batch_X, batch_y


We can add our ``augment_batch`` function to our batch extraction pipeline by invoking the
:py:meth:`~.AbstractDataSource.map` method like so:

.. code:: python

    # Construct an array data source from ``train_X`` and ``train_y``
    ds = data_source.ArrayDataSource([train_X, train_y])

    # Apply augmentation using `augment_batch`
    ds = ds.map(augment_batch)

    # Drawing batches of 64 elements in random order
    for (batch_X, batch_y) in ds.batch_iterator(
            batch_size=64, shuffle=np.random.RandomState(12345)):
        # Processes batches here...

