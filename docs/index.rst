.. batchup documentation master file, created by
   sphinx-quickstart on Sat Jan 20 21:36:31 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

BatchUp
=======

BatchUp is a lightweight Python library for extracting mini-batches of data for the purpose of training neural networks.

A quick example:

.. code:: python

   from batchup import data_source

   # Construct an array data source
   ds = data_source.ArrayDataSource([train_X, train_y])

   # Iterate over samples, drawing batches of 64 elements in
   # random order
   for (batch_X, batch_y) in ds.batch_iterator(batch_size=64,
                                               shuffle=True):
       # Processes batches here...



User Guide
----------

.. toctree::
   :maxdepth: 2

   user/install
   user/basic_batch_iter
   user/data_augmentation
   user/non_array_batch_iter
   user/parallel_batch
   user/composite_data_source
   user/sample_weighting
   :caption: Contents:


API Reference
-------------

.. toctree::
  :maxdepth: 2

  modules/data_source
  modules/sampling
  modules/work_pool


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _GitHub: https://github.com/Britefury/batchup