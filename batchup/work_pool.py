import sys
import numpy as np
import collections
import multiprocessing.managers
import cPickle

import joblib

from . import data_source

SharedConstant = collections.namedtuple('SharedConstant', ['value'])
_SharedRef = collections.namedtuple('_SharedRef', ['key'])

_MAX_PRIMTIVE_ARG_SIZE = 16384


def _deserialise_args(shared_objects, local_objects,
                      serialised_args):  # pragma: no cover
    args = []
    for arg in serialised_args:
        if isinstance(arg, _SharedRef):
            key = arg.key
            if key in local_objects:
                x = local_objects[key]
            else:
                x = cPickle.loads(shared_objects[arg.key])
                local_objects[arg.key] = x
        else:
            x = arg
        args.append(x)
    return tuple(args)


# pragma: no cover
def _serialise_args(shared_objects, args):  # pragma: no cover
    serialised_args = []
    for arg in args:
        if isinstance(arg, SharedConstant):
            value = arg.value
            key = id(value)
            if key not in shared_objects:
                shared_objects[key] = cPickle.dumps(value)
            ref = _SharedRef(key=key)
            serialised = ref
        else:
            serialised = arg
        serialised_args.append(serialised)
    return tuple(serialised_args)


def _apply_async_helper(shared_objects, fn,
                        *serialised_args):  # pragma: no cover
    try:
        try:
            local_objects = getattr(_apply_async_helper, '__local_objects')
        except AttributeError:
            local_objects = {}
            setattr(_apply_async_helper, '__local_objects', local_objects)

        args = _deserialise_args(shared_objects, local_objects,
                                 serialised_args)

        return fn(*args)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise


class WorkerPool(object):
    """
    Create a pool of worker processes.

    Call the `work_stream` method to create a `WorkStream` instance that can
    be used to perform tasks in a separate process.

    The work stream is provided a generator that generates tasks that are to
    be executed in a pool of processes. The work stream will attempt to
    ensure that a buffer of results from those tasks is kept full; retrieving
    a result will cause the work stream to top up the result buffer as
    necessary.
    """

    def __init__(self, processes=1):
        """
        Constructor

        :param processes: (default=1) number of processes to start
        """
        self.__manager = multiprocessing.managers.SyncManager()
        self.__manager.start()
        self.__shared_objects = self.__manager.dict({})
        self.__pool = joblib.pool.MemmapingPool(processes=processes)

    def _apply_async(self, fn, args):
        serialised_args = _serialise_args(self.__shared_objects, args)
        return self.__pool.apply_async(
            _apply_async_helper,
            (self.__shared_objects, fn) + serialised_args)

    def work_stream(self, task_generator, task_buffer_size=20):
        """
        Create a work stream. Supply a task generator that will be used to
        generate tasks to execute in the worker processes.
        Note that creating multiple work streams from the same `WorkerPool`
        will result in all of the work streams sharing the same pool of
        processes; long-running tasks from one work stream will delay other
        work streams that use the same pool.

        >>> def do_some_work(x, y):
        ...     return x.sum() * y.sum()

        >>> def task_generator():
        ...     for i in range(10):
        ...         a = np.random.normal(size=(10,))
        ...         b = np.random.uniform(size=(10,))
        ...         yield do_some_work, (a, b)
        ...
        >>> pool = WorkerPool()
        ... ws = pool.work_stream(task_generator())
        ...
        ... for x in ws.retrieve_iter():
        ...     # `do_some_work` will be invoked in a separate process
        ...     pass

        :param task_generator: a generator function if the form
            `task_generator() -> iterator` that yields tasks in the form of
            tuples that can be passed to `Pool.apply_async` methods; `(fn,)`
            or `(fn, args)` or `(fn, args, kwargs)` where `fn` is a function
            that is to be executed in a worker process,
        :param task_buffer_size: (default=20) the size of the buffer; the
            work stream will try to ensure that this number of tasks are
            awaiting completion
        :return: a `WorkStream` instance
        """
        return WorkStream(self, task_generator,
                          task_buffer_size=task_buffer_size)

    def parallel_data_source(self, ds, batch_buffer_size=20):
        """
        Create a data source whose `batch_iterator` method uses multiple
        processes to generate mini-batches in parallel.

        NOTE: `data_source` should be a random access data source.


         (see
        `data_source` module); batch iterators cannot be passed here.
        ALSO NOTE: the object that is passed in the parameter `data_source`
        SHOULD BE LIGHTWEIGHT as it is passed to the child processes via
        serialization/deserialization, so you most probably *don't* want to
        pass large NumPy arrays here. The recommended approach is to pass
        file names or other information that can be used to find locate large
        sources of data.

        >>> class ExpensiveSource (object):
        ...     def __len__(self):
        ...         return 100
        ...
        ...     def __getitem__(self, indices):
        ...         rng = np.random.RandomState(int(indices.sum()))
        ...         return rng.normal(size=(indices.shape[0], 3))
        ...
        >>> class ExpensiveTarget (object):
        ...     def __len__(self):
        ...         return 100
        ...
        ...     def __getitem__(self, indices):
        ...         rng = np.random.RandomState(int(indices.sum()))
        ...         return rng.randint(low=0, high=10,
        ...                            size=(indices.shape[0],))
        ...
        >>> pool = WorkerPool()
        ...
        ... ds = data_source.ArrayDataSource([ExpensiveSource(),
        ...     ExpensiveTarget()])
        ...
        ... pds = pool.parallel_data_source(ds)
        ...
        ... for (batch_src, batch_tgt) in pds.batch_iterator(batch_size=20):
        ...     # Batch generation will occur in separate processes
        ...     pass

        Parameters
        ----------
        ds: AbstractDataSource where `ds.is_random_access` is `True`
            The data source from which to acquire data

        batch_buffer_size: [optional] int
            The number of mini-batches that will be buffered up to ensure
            that data is always ready when requested

        Returns
        -------
        A `ParallelDataSource` instance.
        """
        return ParallelDataSource(ds, batch_buffer_size, self)


def _pds_extract_helper(data, batch_indices):  # pragma: no cover
    return data.samples_by_indices(batch_indices)


class ParallelDataSource(data_source.AbstractDataSource):
    def __init__(self, ds, batch_buffer_size, pool):
        if not isinstance(ds, data_source.AbstractDataSource):
            raise TypeError(
                'ds must implement `data_source.AbstractDataSource`')
        if not ds.is_random_access:
            raise TypeError('ds must be random access; it is of type '
                            '{}'.format(type(ds)))
        self.__ds = ds
        self.__batch_buffer_size = batch_buffer_size
        self.__pool = pool

    def num_samples(self, **kwargs):
        return self.__ds.num_samples(**kwargs)

    def batch_iterator(self, batch_size, **kwargs):
        def task_generator():
            for ndx_batch in self.__ds.batch_indices_iterator(
                    batch_size, **kwargs):
                yield _pds_extract_helper, (SharedConstant(self.__ds),
                                            ndx_batch)

        ws = self.__pool.work_stream(
            task_generator(), task_buffer_size=self.__batch_buffer_size)
        return ws.retrieve_iter()


class WorkStream(object):
    """
    A work stream, normally constructed using the `WorkerPool.work_stream`
    method.
    """

    def __init__(self, worker_pool, task_generator, task_buffer_size=20):
        assert isinstance(worker_pool, WorkerPool)
        self.__task_gen = task_generator
        self.__buffer_size = task_buffer_size
        self.__result_buffer = collections.deque()
        self.__worker_pool = worker_pool
        self.__populate_buffer()

    def __populate_buffer(self):
        while len(self.__result_buffer) < self.__buffer_size:
            if not self.__enqueue():
                break

    def __enqueue(self):
        try:
            task = self.__task_gen.next()
        except StopIteration:
            return False
        else:
            future = self.__worker_pool._apply_async(*task)
            self.__result_buffer.append(future)
            return True

    def retrieve(self):
        """
        Retrieve a result from executing a task. Note that tasks are executed
        in order and that if the next task has not yet completed, this call
        will block until the result is available.
        :return: the result returned by the task function.
        """
        if len(self.__result_buffer) > 0:
            res = self.__result_buffer.popleft()
            value = res.get()
        else:
            return None

        self.__populate_buffer()

        return value

    def retrieve_iter(self):
        """
        Retrieve a result from executing a task. Note that tasks are executed
        in order and that if the next task has not yet completed, this call
        will block until the result is available.
        :return: the result returned by the task function.
        """
        while len(self.__result_buffer) > 0:
            res = self.__result_buffer.popleft()
            value = res.get()
            self.__populate_buffer()
            yield value
