Sample weighting to alter likelihood of samples
===============================================

BatchUp defines *samplers* that are used to generate the indices of samples that should be combined to form
a mini-batch. They are defined in the :py:mod:`.sampling` module.

When constructing a data source (e.g. :py:class:`.ArrayDataSource`) you can provide a sampler
that will control how the samples are selected.

Buy default one of the standard samplers (:py:class:`.StandardSampler` or :py:class:`.SubsetSampler`)
will be constructed if you don't provide one.

.. contents::



The weighted sampler
--------------------

If you want some samples to be drawn more frequently than others, construct a :py:class:`.WeightedSampler` and pass
it as the sampler argument to the py:class:`.ArrayDataSource` constructor. In the example the per-sample
weights are stored in ``train_w``.

.. code:: python

    from batchup import sampling

    sampler = sampling.WeightedSampler(weights=train_w)

    ds = data_source.ArrayDataSource([train_X, train_y], sampler=sampler)

    # Drawing batches of 64 elements in random order
    for (batch_X, batch_y) in ds.batch_iterator(
            batch_size=64, shuffle=np.random.RandomState(12345)):
        # Processes batches here...

Note that in-order is NOT supported when using :py:class:`.WeightedSampler`, so ``shuffle`` cannot be ``False``
or ``None``.

To draw from a subset of the dataset, use :py:class:`.WeightedSubsetSampler`:

.. code:: python

    from batchup import sampling

    # NOTE that the weights parameter is called `sub_weights` (rather
    # than `weights`) and that it must have the same length as `indices`.
    sampler = sampling.WeightedSubsetSampler(sub_weights=train_w[subset_a],
                                             indices=subset_a)

    ds = data_source.ArrayDataSource([train_X, train_y], sampler=sampler)

    # Drawing batches of 64 elements in random order
    for (batch_X, batch_y) in ds.batch_iterator(
            batch_size=64, shuffle=np.random.RandomState(12345)):
        # Processes batches here...

Counteracting class imbalance
-----------------------------

An alternate constructor method :py:meth:`.WeightedSampler.class_balancing_sampler` is available to construct
a weighted sampler to compensate for class imbalance:

.. code:: python

    # Construct the sampler; NOTE that the `n_classes` argument
    # is *optional*
    sampler = sampling.WeightedSampler.class_balancing_sampler(
        y=train_y, n_classes=train_y.max() + 1)

    ds = data_source.ArrayDataSource([train_X, train_y], sampler=sampler)

    # Drawing batches of 64 elements in random order
    for (batch_X, batch_y) in ds.batch_iterator(
            batch_size=64, shuffle=np.random.RandomState(12345)):
        # Processes batches here...


The :py:meth:`.WeightedSampler.class_balancing_sample_weights` helper method constructs an array of sample
weights in case you wish to modify the weights first:

.. code:: python

    weights = sampling.WeightedSampler.class_balancing_sample_weights(
        y=train_y, n_classes=train_y.max() + 1)

    # Assume `modify_weights` is defined above
    weights = modify_weights(weights)

    # Construct the sampler and the data source
    sampler = sampling.WeightedSampler(weights=weights)
    ds = data_source.ArrayDataSource([train_X, train_y], sampler=sampler)

    # Drawing batches of 64 elements in random order
    for (batch_X, batch_y) in ds.batch_iterator(
            batch_size=64, shuffle=np.random.RandomState(12345)):
        # Processes batches here...
