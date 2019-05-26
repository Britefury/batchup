Parallel processing for faster batching
=======================================

Batch generation pipelines and involve reading images from disk on the fly and applying potentially expensive
data augmentation operations can incur significant overhead. Running such a pipeline in a single threaded/process
along side the training code can result in the GPU spending a significant amount of time waiting for the CPU
to provide it with data.

BatchUp provides worker pools that run the batch generation pipeline in background processes or threads,
ameliorating this problem.

.. contents::


Using threads
-------------

Let's build on the :ref:`array-like-objects` example in which we load the
`Kaggle Dogs vs Cats <https://www.kaggle.com/c/dogs-vs-cats>`_ training set. Let's assume that
``train_X`` references an instance of the ``NonUniformImageFileAccessor`` class defined in that example,
``train_y`` contains ground truth classes and that we have the augmentation function ``augment_batch``:


.. code:: python

    from batchup import data_source, work_pool

    # Build a pool of 4 worker threads:
    th_pool = work_pool.WorkerThreadPool(processes=4)

    # Build our data source and apply augmentation function
    ds = data_source.ArrayDataSource([train_X, train_y])

    ds = ds.map(augment_batch)

    # Construct a parallel data source that prepares mini-batches in the
    # background. It wraps the existing data source `ds` and will try
    # to keep a buffer of 32 mini-batches full to eliminate latency:
    par_ds = th_pool.parallel_data_source(ds, batch_buffer_size=32)

    # As soon as we create an iterator, it will start filling its
    # buffer; lets create an iterator right now to get it going in
    # the background:
    par_iter = par_ds.batch_iterator(
        batch_size=64, shuffle=np.random.RandomState(12345))

    # Do some other initialisation stuff that may take a while...

    # By now, with any luck, some batches will be ready to retrieve

    for (batch_X, batch_y) in par_iter:
        # Process batches here...


Using processes
---------------

In some cases the data source that you wish to parallelize may not be thread safe or may perform poorly in a
multi-threaded environment due to Pythons' Global Interpreter Lock (GIL). In such cases you can use process based
pools that use separate processes rather than threads.

There are two issues that you should be aware of when using processes:

- Process-based pools incur a higher overhead due to having to copy batches between processes. We use
  ``MemmappingPool`` from ``joblib`` to attempt to ameliorate this, but that mechanism only works with NumPy arrays,
  so some pickling will still be performed
- Any classes or functions used -- that includes custom accessor classes and data augmentation functions -- must be
  declared in the top level of a module so that pickle can find them for the purpose of passing references to them
  to the worker processes

That aside, we need only replace the reference to :py:class:`.WorkerThreadPool` with :py:class:`.WorkerProcessPool`:

.. code:: python

    # Build a pool of 4 worker processes:
    proc_pool = work_pool.WorkerProcessPool(processes=4)

    # Construct a data source that prepares mini-batches in the
    # background. It wraps the existing data source `ds` and will try
    # to keep a buffer of 32 mini-batches full to eliminate latency:
    par_ds = proc_pool.parallel_data_source(ds, batch_buffer_size=32)

    # ... use `par_ds` the same way as before ...