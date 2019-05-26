Iterating over data sets of different sizes (e.g. for semi-supervised learning)
===============================================================================

There are some instances where we wish to draw samples from two data sets simultaneously, where the data sets
have different sizes. One example is in semi-supervised learning where we have a small dataset of labeled samples
``lab_X`` with ground truths ``lab_y`` and a larger set of unlabeled samples ``unlab_X``. Lets say we want a
single epoch to consist of the entire unlabeled dataset while looping over the labeled dataset repeatly. The
:py:class:`.data_source.CompositeDataSource` class can help us here.

Without using :py:class:`.CompositeDataSource` we would need the following:

.. code:: python

    rng = np.random.RandomState(12345)

    # Construct the data sources; the labeled data source will
    # repeat infinitely
    ds_lab = data_source.ArrayDataSource([lab_X, lab_y], repeats=-1)
    ds_unlab = data_source.ArrayDataSource([unlab_X])

    # Construct an iterator to get samples from our labeled data source:
    lab_iter = ds_lab.batch_iterator(batch_size=64, shuffle=rng)

    # Iterate over the unlabeled data set in the for-loop
    for (batch_unlab_X,) in ds_unlab.batch_iterator(
            batch_size=64, shuffle=rng):
        # Extract batches from the labeled iterator ourselves
        batch_lab_X, batch_lab_y = next(lab_iter)

        # (we could also use `zip`)

        # Process batches here...

We can use :py:class:`.CompositeDataSource` to simplify the above code. It will drawn samples from both
``ds_lab`` and ``ds_unlab`` and will shuffle the samples from these data source idependently:

.. code:: python

    # Construct the data sources; the labeled data source will
    # repeat infinitely
    ds_lab = data_source.ArrayDataSource([lab_X, lab_y], repeats=-1)
    ds_unlab = data_source.ArrayDataSource([unlab_X])
    # Combine with a `CompositeDataSource`
    ds = data_source.CompositeDataSource([ds_lab, ds_unlab])

    # Iterate over both the labeled and unlabeled samples:
    for (batch_lab_X, batch_lab_y, batch_unlab_X) in ds.batch_iterator(
            batch_size=64, shuffle=rng):
        # Process batches here...

You can also have :py:class:`.CompositeDataSource` generate structured mini-batches that reflect the
structure of the component data sources. The batches will have a nested tuple structure:

.. code:: python

    # Disable flattening with `flatten=False`
    ds_struct = data_source.CompositeDataSource(
        [ds_lab, ds_unlab], flatten=False)

    # Iterate over both the labeled and unlabeled samples.
    for ((batch_lab_X, batch_lab_y), (batch_unlab_X,)) in \
            ds_struct.batch_iterator(batch_size=64, shuffle=rng):
        # Process batches here...