import pytest
import numpy as np


# Helper function to test the callable protocol
def make_batch_iterator_callable(*ds):
    from batchup import data_source

    def batch_iterator(batch_size, shuffle=None):
        # Make `data_source.ArrayDataSource.batch_iterator` do the work :)
        return data_source.ArrayDataSource(list(ds)).batch_iterator(
            batch_size, shuffle=shuffle)
    return batch_iterator


def test_length_of_batch():
    from batchup import data_source

    X = np.arange(10)
    Y = np.arange(20)

    assert data_source._length_of_batch(X) == 10

    assert data_source._length_of_batch((X,)) == 10
    assert data_source._length_of_batch((X, Y)) == 10
    assert data_source._length_of_batch(((X,), Y)) == 10
    assert data_source._length_of_batch((Y, X)) == 20
    assert data_source._length_of_batch(((Y,), X)) == 20

    assert data_source._length_of_batch([X]) == 1
    assert data_source._length_of_batch([X, Y]) == 2
    assert data_source._length_of_batch([[X], Y]) == 2


def test_trim_batch():
    from batchup import data_source

    X = np.arange(10)
    Y = np.arange(20)

    assert (data_source._trim_batch(X, 5) == X[:5]).all()

    b = data_source._trim_batch((X, Y), 5)
    assert isinstance(b, tuple)
    assert (b[0] == X[:5]).all()
    assert (b[1] == Y[:5]).all()

    b = data_source._trim_batch(((X, Y), X, Y), 5)
    assert isinstance(b, tuple)
    assert (b[0][0] == X[:5]).all()
    assert (b[0][1] == Y[:5]).all()
    assert (b[1] == X[:5]).all()
    assert (b[2] == Y[:5]).all()

    b = data_source._trim_batch(([X, Y], X, Y), 5)
    assert isinstance(b, tuple)
    assert (b[0][0] == X).all()
    assert (b[0][1] == Y).all()
    assert (b[1] == X[:5]).all()
    assert (b[2] == Y[:5]).all()


def test_AbstractDataSource():
    from batchup import data_source

    ds = data_source.AbstractDataSource()

    assert not ds.is_random_access

    with pytest.raises(NotImplementedError):
        _ = ds.num_samples()

    with pytest.raises(NotImplementedError):
        ds.batch_iterator(256)


def test_RandomAccessDataSource():
    from batchup import data_source

    ds = data_source.RandomAccessDataSource(length=10)

    assert ds.is_random_access

    # `samples_by_indices_nomapping` should be abstract
    with pytest.raises(NotImplementedError):
        _ = ds.samples_by_indices_nomapping(slice(None))

    # Test index generation

    # In order
    ndx_iter = ds.batch_indices_iterator(batch_size=5)
    all = np.arange(10)
    batches = list(ndx_iter)
    assert len(batches) == 2
    assert (all[batches[0]] == np.arange(0, 5)).all()
    assert (all[batches[1]] == np.arange(5, 10)).all()

    # In order, last batch short
    ndx_iter = ds.batch_indices_iterator(batch_size=6)
    all = np.arange(10)
    batches = list(ndx_iter)
    assert len(batches) == 2
    assert (all[batches[0]] == np.arange(0, 6)).all()
    assert (all[batches[1]] == np.arange(6, 10)).all()

    # Shuffled
    shuffled_ndx_iter = ds.batch_indices_iterator(
        batch_size=5, shuffle=np.random.RandomState(12345))
    batches = list(shuffled_ndx_iter)
    order = np.random.RandomState(12345).permutation(10)
    assert len(batches) == 2
    assert (all[batches[0]] == order[0:5]).all()
    assert (all[batches[1]] == order[5:10]).all()

    # Shuffled, last batch short
    shuffled_ndx_iter = ds.batch_indices_iterator(
        batch_size=6, shuffle=np.random.RandomState(12345))
    batches = list(shuffled_ndx_iter)
    order = np.random.RandomState(12345).permutation(10)
    assert len(batches) == 2
    assert (all[batches[0]] == order[0:6]).all()
    assert (all[batches[1]] == order[6:10]).all()


def test_RandomAccessDataSource_indices():
    from batchup import data_source

    indices = np.random.RandomState(12345).permutation(20)[:10]
    ds = data_source.RandomAccessDataSource(length=20, indices=indices)

    # Test index generation

    # In order
    ndx_iter = ds.batch_indices_iterator(batch_size=5)
    all = np.arange(20)
    batches = list(ndx_iter)
    assert len(batches) == 2
    assert (all[batches[0]] == indices[0:5]).all()
    assert (all[batches[1]] == indices[5:10]).all()

    # In order, last batch short
    ndx_iter = ds.batch_indices_iterator(batch_size=6)
    all = np.arange(20)
    batches = list(ndx_iter)
    assert len(batches) == 2
    assert (all[batches[0]] == indices[0:6]).all()
    assert (all[batches[1]] == indices[6:10]).all()

    # Shuffled
    shuffled_ndx_iter = ds.batch_indices_iterator(
        batch_size=5, shuffle=np.random.RandomState(12345))
    batches = list(shuffled_ndx_iter)
    order = np.random.RandomState(12345).permutation(10)
    assert len(batches) == 2
    assert (all[batches[0]] == indices[order[0:5]]).all()
    assert (all[batches[1]] == indices[order[5:10]]).all()

    # Shuffled, last batch short
    shuffled_ndx_iter = ds.batch_indices_iterator(
        batch_size=6, shuffle=np.random.RandomState(12345))
    batches = list(shuffled_ndx_iter)
    order = np.random.RandomState(12345).permutation(10)
    assert len(batches) == 2
    assert (all[batches[0]] == indices[order[0:6]]).all()
    assert (all[batches[1]] == indices[order[6:10]]).all()


def test_RandomAccessDataSource_repeated():
    from batchup import data_source

    ds_3 = data_source.RandomAccessDataSource(
        length=10, repeats=3)

    # Test index generation

    # In order
    ndx_iter = ds_3.batch_indices_iterator(batch_size=5)
    all = np.arange(10)
    batches = list(ndx_iter)
    assert len(batches) == 6
    assert (all[batches[0]] == np.arange(0, 5)).all()
    assert (all[batches[1]] == np.arange(5, 10)).all()
    assert (all[batches[2]] == np.arange(0, 5)).all()
    assert (all[batches[3]] == np.arange(5, 10)).all()
    assert (all[batches[4]] == np.arange(0, 5)).all()
    assert (all[batches[5]] == np.arange(5, 10)).all()

    # In order, last batch short
    ndx_iter = ds_3.batch_indices_iterator(batch_size=7)
    all = np.arange(10)
    batches = list(ndx_iter)
    assert len(batches) == 5
    assert (all[batches[0]] == np.arange(0, 7)).all()
    assert (all[batches[1]] == np.append(np.arange(7, 10),
                                         np.arange(0, 4))).all()
    assert (all[batches[2]] == np.append(np.arange(4, 10),
                                         np.arange(0, 1))).all()
    assert (all[batches[3]] == np.arange(1, 8)).all()
    assert (all[batches[4]] == np.arange(8, 10)).all()

    # Shuffled
    shuffled_ndx_iter = ds_3.batch_indices_iterator(
        batch_size=5, shuffle=np.random.RandomState(12345))
    batches = list(shuffled_ndx_iter)
    order_rng = np.random.RandomState(12345)
    order = np.concatenate([order_rng.permutation(10) for _ in range(3)])
    assert len(batches) == 6
    assert (all[batches[0]] == order[0:5]).all()
    assert (all[batches[1]] == order[5:10]).all()
    assert (all[batches[2]] == order[10:15]).all()
    assert (all[batches[3]] == order[15:20]).all()
    assert (all[batches[4]] == order[20:25]).all()
    assert (all[batches[5]] == order[25:30]).all()

    # Shuffled, last batch short
    shuffled_ndx_iter = ds_3.batch_indices_iterator(
        batch_size=7, shuffle=np.random.RandomState(12345))
    batches = list(shuffled_ndx_iter)
    order_rng = np.random.RandomState(12345)
    order = np.concatenate([order_rng.permutation(10) for _ in range(3)])
    assert len(batches) == 5
    assert (all[batches[0]] == order[0:7]).all()
    assert (all[batches[1]] == order[7:14]).all()
    assert (all[batches[2]] == order[14:21]).all()
    assert (all[batches[3]] == order[21:28]).all()
    assert (all[batches[4]] == order[28:30]).all()


def test_RandomAccessDataSource_indices_repeated():
    from batchup import data_source

    indices = np.random.RandomState(12345).permutation(20)[:10]
    ds_3 = data_source.RandomAccessDataSource(
        length=20, indices=indices, repeats=3)

    # Test index generation

    # In order
    ndx_iter = ds_3.batch_indices_iterator(batch_size=5)
    all = np.arange(20)
    batches = list(ndx_iter)
    assert len(batches) == 6
    assert (all[batches[0]] == indices[0:5]).all()
    assert (all[batches[1]] == indices[5:10]).all()
    assert (all[batches[2]] == indices[0:5]).all()
    assert (all[batches[3]] == indices[5:10]).all()
    assert (all[batches[4]] == indices[0:5]).all()
    assert (all[batches[5]] == indices[5:10]).all()

    # In order, last batch short
    ndx_iter = ds_3.batch_indices_iterator(batch_size=7)
    all = np.arange(20)
    batches = list(ndx_iter)
    assert len(batches) == 5
    assert (all[batches[0]] == indices[0:7]).all()
    assert (all[batches[1]] == np.append(indices[7:10],
                                         indices[0:4])).all()
    assert (all[batches[2]] == np.append(indices[4:10],
                                         indices[0:1])).all()
    assert (all[batches[3]] == indices[1:8]).all()
    assert (all[batches[4]] == indices[8:10]).all()

    # Shuffled
    shuffled_ndx_iter = ds_3.batch_indices_iterator(
        batch_size=5, shuffle=np.random.RandomState(12345))
    batches = list(shuffled_ndx_iter)
    order_rng = np.random.RandomState(12345)
    order = np.concatenate([order_rng.permutation(10) for _ in range(3)])
    assert len(batches) == 6
    assert (all[batches[0]] == indices[order[0:5]]).all()
    assert (all[batches[1]] == indices[order[5:10]]).all()
    assert (all[batches[2]] == indices[order[10:15]]).all()
    assert (all[batches[3]] == indices[order[15:20]]).all()
    assert (all[batches[4]] == indices[order[20:25]]).all()
    assert (all[batches[5]] == indices[order[25:30]]).all()

    # Shuffled, last batch short
    shuffled_ndx_iter = ds_3.batch_indices_iterator(
        batch_size=7, shuffle=np.random.RandomState(12345))
    batches = list(shuffled_ndx_iter)
    order_rng = np.random.RandomState(12345)
    order = np.concatenate([order_rng.permutation(10) for _ in range(3)])
    assert len(batches) == 5
    assert (all[batches[0]] == indices[order[0:7]]).all()
    assert (all[batches[1]] == indices[order[7:14]]).all()
    assert (all[batches[2]] == indices[order[14:21]]).all()
    assert (all[batches[3]] == indices[order[21:28]]).all()
    assert (all[batches[4]] == indices[order[28:30]]).all()


def test_RandomAccessDataSource_repeated_small_dataset():
    from batchup import data_source

    ds_inf = data_source.RandomAccessDataSource(
        length=20, repeats=-1)

    # Test index generation

    # In order
    ndx_iter = ds_inf.batch_indices_iterator(batch_size=64)
    all = np.arange(20)
    batches = [next(ndx_iter) for _ in range(10)]
    assert len(batches) == 10
    x = np.concatenate([all[b] for b in batches], axis=0)
    assert (x == np.tile(np.arange(20), [32])).all()

    # Shuffled
    shuffled_ndx_iter = ds_inf.batch_indices_iterator(
        batch_size=64, shuffle=np.random.RandomState(12345))
    batches = [next(shuffled_ndx_iter) for _ in range(10)]
    order_rng = np.random.RandomState(12345)
    order = np.concatenate([order_rng.permutation(20) for _ in range(32)])
    assert len(batches) == 10
    x = np.concatenate([all[b] for b in batches], axis=0)
    assert (x == order).all()

    # Test fixed number of repetitions
    ds_16 = data_source.RandomAccessDataSource(
        length=20, repeats=16)
    assert ds_16.num_samples() == 320

    # In order
    ndx_iter = ds_16.batch_indices_iterator(batch_size=64)
    all = np.arange(20)
    batches = list(ndx_iter)
    assert len(batches) == 5
    x = np.concatenate([all[b] for b in batches], axis=0)
    assert (x == np.tile(np.arange(20), [16])).all()

    # Shuffled
    shuffled_ndx_iter = ds_16.batch_indices_iterator(
        batch_size=64, shuffle=np.random.RandomState(12345))
    batches = list(shuffled_ndx_iter)
    order_rng = np.random.RandomState(12345)
    order = np.concatenate([order_rng.permutation(20) for _ in range(16)])
    assert len(batches) == 5
    x = np.concatenate([all[b] for b in batches], axis=0)
    assert (x == order).all()


def test_RandomAccessDataSource_indices_repeated_small_dataset():
    from batchup import data_source

    indices = np.random.RandomState(12345).permutation(40)[:20]
    ds_inf = data_source.RandomAccessDataSource(
        length=40, indices=indices, repeats=-1)

    # Test index generation

    # In order
    ndx_iter = ds_inf.batch_indices_iterator(batch_size=64)
    all = np.arange(40)
    batches = [next(ndx_iter) for _ in range(10)]
    assert len(batches) == 10
    x = np.concatenate([all[b] for b in batches], axis=0)
    assert (x == np.tile(indices, [32])).all()

    # Shuffled
    shuffled_ndx_iter = ds_inf.batch_indices_iterator(
        batch_size=64, shuffle=np.random.RandomState(12345))
    batches = [next(shuffled_ndx_iter) for _ in range(10)]
    order_rng = np.random.RandomState(12345)
    order = np.concatenate([order_rng.permutation(20) for _ in range(32)])
    assert len(batches) == 10
    x = np.concatenate([all[b] for b in batches], axis=0)
    assert (x == indices[order]).all()

    # Test fixed number of repetitions
    ds_16 = data_source.RandomAccessDataSource(
        length=40, indices=indices, repeats=16)
    assert ds_16.num_samples() == 320

    # In order
    ndx_iter = ds_16.batch_indices_iterator(batch_size=64)
    all = np.arange(40)
    batches = list(ndx_iter)
    assert len(batches) == 5
    x = np.concatenate([all[b] for b in batches], axis=0)
    assert (x == np.tile(indices, [16])).all()

    # Shuffled
    shuffled_ndx_iter = ds_16.batch_indices_iterator(
        batch_size=64, shuffle=np.random.RandomState(12345))
    batches = list(shuffled_ndx_iter)
    order_rng = np.random.RandomState(12345)
    order = np.concatenate([order_rng.permutation(20) for _ in range(16)])
    assert len(batches) == 5
    x = np.concatenate([all[b] for b in batches], axis=0)
    assert (x == indices[order]).all()


def test_ArrayDataSource():
    from batchup import data_source, sampling

    # Test `len(ds)`
    a3a = np.arange(3)
    a3b = np.arange(6).reshape((3, 2))
    a10 = np.arange(10)

    assert data_source.ArrayDataSource([a3a]).num_samples() == 3
    assert data_source.ArrayDataSource([a10]).num_samples() == 10
    assert data_source.ArrayDataSource([a3a, a3b]).num_samples() == 3

    X = np.arange(45)
    Y = np.arange(90).reshape((45, 2))
    ads = data_source.ArrayDataSource([X, Y])

    # Test `samples_by_indices_nomapping`
    batch = ads.samples_by_indices_nomapping(np.arange(15))
    assert (batch[0] == X[:15]).all()
    assert (batch[1] == Y[:15]).all()

    # Test `samples_by_indices`
    batch = ads.samples_by_indices(np.arange(15))
    assert (batch[0] == X[:15]).all()
    assert (batch[1] == Y[:15]).all()

    # Test `batch_iterator`

    # Three in-order batches
    batches = list(ads.batch_iterator(batch_size=15))
    # Three batches
    assert len(batches) == 3
    # Two items in each batch
    assert len(batches[0]) == 2
    assert len(batches[1]) == 2
    assert len(batches[2]) == 2
    # Verify values
    assert (batches[0][0] == X[:15]).all()
    assert (batches[0][1] == Y[:15]).all()
    assert (batches[1][0] == X[15:30]).all()
    assert (batches[1][1] == Y[15:30]).all()
    assert (batches[2][0] == X[30:]).all()
    assert (batches[2][1] == Y[30:]).all()

    # Ensure that shuffle=False results in three in-order batches
    batches = list(ads.batch_iterator(batch_size=15, shuffle=False))
    # Three batches
    assert len(batches) == 3
    # Two items in each batch
    assert len(batches[0]) == 2
    assert len(batches[1]) == 2
    assert len(batches[2]) == 2
    # Verify values
    assert (batches[0][0] == X[:15]).all()
    assert (batches[0][1] == Y[:15]).all()
    assert (batches[1][0] == X[15:30]).all()
    assert (batches[1][1] == Y[15:30]).all()
    assert (batches[2][0] == X[30:]).all()
    assert (batches[2][1] == Y[30:]).all()

    # Three shuffled batches
    batches = list(ads.batch_iterator(
        batch_size=15, shuffle=np.random.RandomState(12345)))
    # Get the expected order
    order = np.random.RandomState(12345).permutation(45)
    # Three batches
    assert len(batches) == 3
    # Two items in each batch
    assert len(batches[0]) == 2
    assert len(batches[1]) == 2
    assert len(batches[2]) == 2
    # Verify values
    assert (batches[0][0] == X[order[:15]]).all()
    assert (batches[0][1] == Y[order[:15]]).all()
    assert (batches[1][0] == X[order[15:30]]).all()
    assert (batches[1][1] == Y[order[15:30]]).all()
    assert (batches[2][0] == X[order[30:]]).all()
    assert (batches[2][1] == Y[order[30:]]).all()

    # Check that shuffle=True uses NumPy's default RNG
    np.random.seed(12345)
    batches = list(ads.batch_iterator(batch_size=15, shuffle=True))
    # Get the expected order
    order = np.random.RandomState(12345).permutation(45)
    # Three batches
    assert len(batches) == 3
    # Two items in each batch
    assert len(batches[0]) == 2
    assert len(batches[1]) == 2
    assert len(batches[2]) == 2
    # Verify values
    assert (batches[0][0] == X[order[:15]]).all()
    assert (batches[0][1] == Y[order[:15]]).all()
    assert (batches[1][0] == X[order[15:30]]).all()
    assert (batches[1][1] == Y[order[15:30]]).all()
    assert (batches[2][0] == X[order[30:]]).all()
    assert (batches[2][1] == Y[order[30:]]).all()

    # Ensure that indices/repeats/sampler parameter sanity check works
    sampler = sampling.StandardSampler(10)
    with pytest.raises(ValueError):
        _ = data_source.ArrayDataSource([X, Y], indices=np.arange(3),
                                        sampler=sampler)

    with pytest.raises(ValueError):
        _ = data_source.ArrayDataSource([X, Y], repeats=2,
                                        sampler=sampler)

    with pytest.raises(ValueError):
        _ = data_source.ArrayDataSource([X, Y], repeats=-1,
                                        sampler=sampler)

    # Check that constructing an array data source given input arrays
    # of differing lengths raises ValueError
    with pytest.raises(ValueError):
        _ = data_source.ArrayDataSource([
            np.arange(20), np.arange(50).reshape((25, 2))])

    # Check that `ArrayDataSource` raises TypeError if the list of arrays
    # is not a list
    with pytest.raises(TypeError):
        _ = data_source.ArrayDataSource(X)


def test_ArrayDataSource_indices():
    from batchup import data_source

    # Test `len(ds)`
    a3a = np.arange(3)
    a3b = np.arange(6).reshape((3, 2))
    a10 = np.arange(10)

    assert data_source.ArrayDataSource([a3a]).num_samples() == 3
    assert data_source.ArrayDataSource([a10]).num_samples() == 10
    assert data_source.ArrayDataSource([a3a, a3b]).num_samples() == 3

    X = np.arange(90)
    Y = np.arange(180).reshape((90, 2))
    indices = np.random.permutation(90)[:45]
    ads = data_source.ArrayDataSource([X, Y], indices=indices)

    # Test `samples_by_indices_nomapping`
    batch = ads.samples_by_indices_nomapping(np.arange(15))
    assert (batch[0] == X[:15]).all()
    assert (batch[1] == Y[:15]).all()

    # Test `samples_by_indices`
    batch = ads.samples_by_indices(np.arange(15))
    assert (batch[0] == X[indices[:15]]).all()
    assert (batch[1] == Y[indices[:15]]).all()

    # Test `batch_iterator`

    # Three in-order batches
    batches = list(ads.batch_iterator(batch_size=15))
    # Three batches
    assert len(batches) == 3
    # Two items in each batch
    assert len(batches[0]) == 2
    assert len(batches[1]) == 2
    assert len(batches[2]) == 2
    # Verify values
    assert (batches[0][0] == X[indices[:15]]).all()
    assert (batches[0][1] == Y[indices[:15]]).all()
    assert (batches[1][0] == X[indices[15:30]]).all()
    assert (batches[1][1] == Y[indices[15:30]]).all()
    assert (batches[2][0] == X[indices[30:]]).all()
    assert (batches[2][1] == Y[indices[30:]]).all()

    # Three shuffled batches
    batches = list(ads.batch_iterator(
        batch_size=15, shuffle=np.random.RandomState(12345)))
    # Get the expected order
    order = np.random.RandomState(12345).permutation(45)
    # Three batches
    assert len(batches) == 3
    # Two items in each batch
    assert len(batches[0]) == 2
    assert len(batches[1]) == 2
    assert len(batches[2]) == 2
    # Verify values
    assert (batches[0][0] == X[indices[order[:15]]]).all()
    assert (batches[0][1] == Y[indices[order[:15]]]).all()
    assert (batches[1][0] == X[indices[order[15:30]]]).all()
    assert (batches[1][1] == Y[indices[order[15:30]]]).all()
    assert (batches[2][0] == X[indices[order[30:]]]).all()
    assert (batches[2][1] == Y[indices[order[30:]]]).all()


def test_ArrayDataSource_repeated():
    from batchup import data_source

    X = np.arange(50)
    Y = np.arange(100).reshape((50, 2))

    # Helper function for checking the resulting mini-batches
    def check_batches(batches, expected_n, order):
        # Eight batches
        assert len(batches) == expected_n
        # Verify contents
        for batch_i, batch in enumerate(batches):
            # Two items in each batch
            assert len(batch) == 2
            # Compute and wrap start and end indices
            start = batch_i * 20
            end = start + 20
            # Get the indices of the expected samples from the `order` array
            if end > start:
                batch_order = order[start:end]
            else:
                batch_order = np.append(order[start:], order[:end], axis=0)
            # Verify values
            assert batch[0].shape[0] == batch_order.shape[0]
            assert batch[1].shape[0] == batch_order.shape[0]
            assert (batch[0] == X[batch_order]).all()
            assert (batch[1] == Y[batch_order]).all()

    # Check size
    assert data_source.ArrayDataSource([X, Y], repeats=1).num_samples() == 50
    assert data_source.ArrayDataSource([X, Y], repeats=2).num_samples() == 100
    assert data_source.ArrayDataSource([X, Y], repeats=5).num_samples() == 250
    inf_ds = data_source.ArrayDataSource([X, Y], repeats=-1)
    assert inf_ds.num_samples() == np.inf

    # 3 repetitions; 150 samples, 8 in-order batches
    ads_3 = data_source.ArrayDataSource([X, Y], repeats=3)
    inorder_iter = ads_3.batch_iterator(batch_size=20)
    batches = list(inorder_iter)
    order = np.concatenate([np.arange(50)] * 3, axis=0)
    check_batches(batches, 8, order)

    # 3 repetitions; 150 samples, 8 shuffled batches
    shuffled_iter = ads_3.batch_iterator(batch_size=20,
                                         shuffle=np.random.RandomState(12345))
    batches = list(shuffled_iter)
    order_shuffle_rng = np.random.RandomState(12345)
    order = np.concatenate(
        [order_shuffle_rng.permutation(50),
         order_shuffle_rng.permutation(50),
         order_shuffle_rng.permutation(50)], axis=0)
    check_batches(batches, 8, order)

    # Infinite repetitions; take 5 in-order batches
    ads_inf = data_source.ArrayDataSource([X, Y], repeats=-1)
    inorder_iter = ads_inf.batch_iterator(batch_size=20)
    batches = [next(inorder_iter) for i in range(5)]
    order = np.concatenate([np.arange(50)] * 2, axis=0)
    check_batches(batches, 5, order)

    # Infinite repetitions; take 5 shuffled batches
    shuffled_iter = ads_inf.batch_iterator(
        batch_size=20, shuffle=np.random.RandomState(12345))
    batches = [next(shuffled_iter) for i in range(5)]
    # Get the expected order
    order_shuffle_rng = np.random.RandomState(12345)
    order = np.append(order_shuffle_rng.permutation(50),
                      order_shuffle_rng.permutation(50), axis=0)
    check_batches(batches, 5, order)

    # Check invalid values for repeats
    with pytest.raises(ValueError):
        data_source.ArrayDataSource([X, Y], repeats=0)

    with pytest.raises(ValueError):
        data_source.ArrayDataSource([X, Y], repeats=-2)


def test_ArrayDataSource_indices_repeated():
    from batchup import data_source

    X = np.arange(100)
    Y = np.arange(200).reshape((100, 2))
    indices = np.random.permutation(100)[:50]

    # Helper function for checking the resulting mini-batches
    def check_batches(batches, expected_n, order):
        # Eight batches
        assert len(batches) == expected_n
        # Verify contents
        for batch_i, batch in enumerate(batches):
            # Two items in each batch
            assert len(batch) == 2
            # Compute and wrap start and end indices
            start = batch_i * 20
            end = start + 20
            # Get the indices of the expected samples from the `order` array
            batch_order = order[start:end]
            # Verify values
            assert batch[0].shape[0] == batch_order.shape[0]
            assert batch[1].shape[0] == batch_order.shape[0]
            assert (batch[0] == X[batch_order]).all()
            assert (batch[1] == Y[batch_order]).all()

    # 3 repetitions; 8 in-order batches
    ads_3 = data_source.ArrayDataSource([X, Y], indices=indices, repeats=3)
    inorder_iter = ads_3.batch_iterator(batch_size=20)
    batches = list(inorder_iter)
    order = np.concatenate([indices, indices, indices], axis=0)
    check_batches(batches, 8, order)

    # 3 repetitions; 8 shuffled batches
    ads_3 = data_source.ArrayDataSource([X, Y], indices=indices, repeats=3)
    inorder_iter = ads_3.batch_iterator(batch_size=20,
                                        shuffle=np.random.RandomState(12345))
    batches = list(inorder_iter)
    # Compute the expected order
    order_shuffle_rng = np.random.RandomState(12345)
    order = np.concatenate(
        [order_shuffle_rng.permutation(indices),
         order_shuffle_rng.permutation(indices),
         order_shuffle_rng.permutation(indices)], axis=0)
    check_batches(batches, 8, order)

    # Infinite repetitions; take 5 in-order batches
    ads_inf = data_source.ArrayDataSource([X, Y], indices=indices, repeats=-1)
    inorder_iter = ads_inf.batch_iterator(batch_size=20)
    batches = [next(inorder_iter) for i in range(5)]
    order = np.concatenate([indices, indices], axis=0)
    check_batches(batches, 5, order)

    # Infinite repetitions; take 5 shuffled batches
    shuffled_iter = ads_inf.batch_iterator(
        batch_size=20, shuffle=np.random.RandomState(12345))
    batches = [next(shuffled_iter) for i in range(5)]
    # Compute the expected order
    order_shuffle_rng = np.random.RandomState(12345)
    order = np.append(order_shuffle_rng.permutation(indices),
                      order_shuffle_rng.permutation(indices), axis=0)
    check_batches(batches, 5, order)


def test_ArrayDataSource_repeated_small_dataset():
    from batchup import data_source

    X = np.arange(20)
    Y = np.arange(40).reshape((20, 2))

    # Helper function for checking the resulting mini-batches
    def check_batches(batches, expected_n, order, batch_size):
        # Eight batches
        assert len(batches) == expected_n
        # Verify contents
        for batch_i, batch in enumerate(batches):
            # Two items in each batch
            assert len(batch) == 2
            # Compute and wrap start and end indices
            start = batch_i * batch_size
            end = start + batch_size
            # Get the indices of the expected samples from the `order` array
            batch_order = order[start:end]
            # Verify values
            assert batch[0].shape[0] == batch_order.shape[0]
            assert batch[1].shape[0] == batch_order.shape[0]
            assert (batch[0] == X[batch_order]).all()
            assert (batch[1] == Y[batch_order]).all()

    # Infinite repetitions; take 10 in-order batches of 64 samples each
    ads_inf = data_source.ArrayDataSource([X, Y], repeats=-1)
    inorder_iter = ads_inf.batch_iterator(batch_size=64)
    batches = [next(inorder_iter) for i in range(10)]
    order = np.concatenate([np.arange(20)] * 32, axis=0)
    check_batches(batches, 10, order, 64)

    # Infinite repetitions; take 10 shuffled batches
    shuffled_iter = ads_inf.batch_iterator(
        batch_size=64, shuffle=np.random.RandomState(12345))
    batches = [next(shuffled_iter) for i in range(10)]
    # Get the expected order
    order_shuffle_rng = np.random.RandomState(12345)
    order = np.concatenate([order_shuffle_rng.permutation(20)
                            for _ in range(32)], axis=0)
    check_batches(batches, 10, order, 64)

    # 16 repetitions; take 5 in-order batches of 64 samples each
    ads_16 = data_source.ArrayDataSource([X, Y], repeats=16)
    inorder_iter = ads_16.batch_iterator(batch_size=64)
    batches = list(inorder_iter)
    order = np.concatenate([np.arange(20)] * 16, axis=0)
    check_batches(batches, 5, order, 64)

    # 16 repetitions; take 5 shuffled batches
    shuffled_iter = ads_16.batch_iterator(
        batch_size=64, shuffle=np.random.RandomState(12345))
    batches = list(shuffled_iter)
    # Get the expected order
    order_shuffle_rng = np.random.RandomState(12345)
    order = np.concatenate([order_shuffle_rng.permutation(20)
                            for _ in range(16)], axis=0)
    check_batches(batches, 5, order, 64)


def test_ArrayDataSource_indices_repeated_small_dataset():
    from batchup import data_source

    X = np.arange(20)
    Y = np.arange(40).reshape((20, 2))
    indices = np.random.permutation(20)[:10]

    # Helper function for checking the resulting mini-batches
    def check_batches(batches, expected_n, order, batch_size):
        # Eight batches
        assert len(batches) == expected_n
        # Verify contents
        for batch_i, batch in enumerate(batches):
            # Two items in each batch
            assert len(batch) == 2
            # Compute and wrap start and end indices
            start = batch_i * batch_size
            end = start + batch_size
            # Get the indices of the expected samples from the `order` array
            batch_order = order[start:end]
            # Verify values
            assert batch[0].shape[0] == batch_order.shape[0]
            assert batch[1].shape[0] == batch_order.shape[0]
            assert (batch[0] == X[batch_order]).all()
            assert (batch[1] == Y[batch_order]).all()

    # Infinite repetitions; take 10 in-order batches
    ads_inf = data_source.ArrayDataSource([X, Y], indices=indices, repeats=-1)
    inorder_iter = ads_inf.batch_iterator(batch_size=64)
    batches = [next(inorder_iter) for i in range(10)]
    order = np.concatenate([indices] * 64, axis=0)
    check_batches(batches, 10, order, 64)

    # Infinite repetitions; take 10 shuffled batches
    shuffled_iter = ads_inf.batch_iterator(
        batch_size=64, shuffle=np.random.RandomState(12345))
    batches = [next(shuffled_iter) for i in range(10)]
    # Compute the expected order
    order_shuffle_rng = np.random.RandomState(12345)
    order = np.concatenate([order_shuffle_rng.permutation(indices)
                            for _ in range(64)], axis=0)
    check_batches(batches, 10, order, 64)


def test_ArrayDataSource_include_indices():
    from batchup import data_source

    # Test `len(ds)`
    a3a = np.arange(3)
    a3b = np.arange(6).reshape((3, 2))
    a10 = np.arange(10)

    assert data_source.ArrayDataSource([a3a]).num_samples() == 3
    assert data_source.ArrayDataSource([a10]).num_samples() == 10
    assert data_source.ArrayDataSource([a3a, a3b]).num_samples() == 3

    X = np.arange(90)
    Y = np.arange(180).reshape((90, 2))
    indices = np.random.permutation(90)[:45]
    ads = data_source.ArrayDataSource([X[:45], Y[:45]], include_indices=True)
    ads_sub = data_source.ArrayDataSource([X, Y], indices=indices,
                                          include_indices=True)

    # Test `samples_by_indices_nomapping`
    batch = ads.samples_by_indices_nomapping(np.arange(15))
    assert (batch[0] == np.arange(15)).all()
    assert (batch[1] == X[:15]).all()
    assert (batch[2] == Y[:15]).all()

    batch = ads_sub.samples_by_indices_nomapping(np.arange(15))
    assert (batch[0] == np.arange(15)).all()
    assert (batch[1] == X[:15]).all()
    assert (batch[2] == Y[:15]).all()

    batch = ads_sub.samples_by_indices_nomapping(slice(0, 15))
    assert (batch[0] == np.arange(15)).all()
    assert (batch[1] == X[:15]).all()
    assert (batch[2] == Y[:15]).all()

    # Test `samples_by_indices`
    batch = ads.samples_by_indices(np.arange(15))
    assert (batch[0] == np.arange(15)).all()
    assert (batch[1] == X[:15]).all()
    assert (batch[2] == Y[:15]).all()

    batch = ads_sub.samples_by_indices(np.arange(15))
    assert (batch[0] == indices[:15]).all()
    assert (batch[1] == X[indices[:15]]).all()
    assert (batch[2] == Y[indices[:15]]).all()

    # Test `batch_iterator`

    # Three in-order batches
    batches = list(ads.batch_iterator(batch_size=15))
    # Three batches
    assert len(batches) == 3
    # Two items in each batch
    assert len(batches[0]) == 3
    assert len(batches[1]) == 3
    assert len(batches[2]) == 3
    # Verify values
    assert (batches[0][0] == np.arange(15)).all()
    assert (batches[0][1] == X[:15]).all()
    assert (batches[0][2] == Y[:15]).all()
    assert (batches[1][0] == np.arange(15, 30)).all()
    assert (batches[1][1] == X[15:30]).all()
    assert (batches[1][2] == Y[15:30]).all()
    assert (batches[2][0] == np.arange(30, 45)).all()
    assert (batches[2][1] == X[30:45]).all()
    assert (batches[2][2] == Y[30:45]).all()

    # Three in-order batches
    batches = list(ads_sub.batch_iterator(batch_size=15))
    # Three batches
    assert len(batches) == 3
    # Two items in each batch
    assert len(batches[0]) == 3
    assert len(batches[1]) == 3
    assert len(batches[2]) == 3
    # Verify values
    assert (batches[0][0] == indices[:15]).all()
    assert (batches[0][1] == X[indices[:15]]).all()
    assert (batches[0][2] == Y[indices[:15]]).all()
    assert (batches[1][0] == indices[15:30]).all()
    assert (batches[1][1] == X[indices[15:30]]).all()
    assert (batches[1][2] == Y[indices[15:30]]).all()
    assert (batches[2][0] == indices[30:]).all()
    assert (batches[2][1] == X[indices[30:]]).all()
    assert (batches[2][2] == Y[indices[30:]]).all()

    # Three shuffled batches
    batches = list(ads.batch_iterator(
        batch_size=15, shuffle=np.random.RandomState(12345)))
    # Get the expected order
    order = np.random.RandomState(12345).permutation(45)
    # Three batches
    assert len(batches) == 3
    # Two items in each batch
    assert len(batches[0]) == 3
    assert len(batches[1]) == 3
    assert len(batches[2]) == 3
    # Verify values
    assert (batches[0][0] == order[:15]).all()
    assert (batches[0][1] == X[order[:15]]).all()
    assert (batches[0][2] == Y[order[:15]]).all()
    assert (batches[1][0] == order[15:30]).all()
    assert (batches[1][1] == X[order[15:30]]).all()
    assert (batches[1][2] == Y[order[15:30]]).all()
    assert (batches[2][0] == order[30:45]).all()
    assert (batches[2][1] == X[order[30:45]]).all()
    assert (batches[2][2] == Y[order[30:45]]).all()

    # Three shuffled batches
    batches = list(ads_sub.batch_iterator(
        batch_size=15, shuffle=np.random.RandomState(12345)))
    # Get the expected order
    order = np.random.RandomState(12345).permutation(45)
    # Three batches
    assert len(batches) == 3
    # Two items in each batch
    assert len(batches[0]) == 3
    assert len(batches[1]) == 3
    assert len(batches[2]) == 3
    # Verify values
    assert (batches[0][0] == indices[order[:15]]).all()
    assert (batches[0][1] == X[indices[order[:15]]]).all()
    assert (batches[0][2] == Y[indices[order[:15]]]).all()
    assert (batches[1][0] == indices[order[15:30]]).all()
    assert (batches[1][1] == X[indices[order[15:30]]]).all()
    assert (batches[1][2] == Y[indices[order[15:30]]]).all()
    assert (batches[2][0] == indices[order[30:]]).all()
    assert (batches[2][1] == X[indices[order[30:]]]).all()
    assert (batches[2][2] == Y[indices[order[30:]]]).all()


def test_CallableDataSource():
    from batchup import data_source

    # Data to extract batches from
    X = np.arange(45)
    Y = np.arange(90).reshape((45, 2))

    # Helper functions to check
    def check_in_order_batches(batches):
        # Three batches
        assert len(batches) == 3
        # Two items in each batch
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        assert len(batches[2]) == 2
        # Verify values
        assert (batches[0][0] == X[:15]).all()
        assert (batches[0][1] == Y[:15]).all()
        assert (batches[1][0] == X[15:30]).all()
        assert (batches[1][1] == Y[15:30]).all()
        assert (batches[2][0] == X[30:]).all()
        assert (batches[2][1] == Y[30:]).all()

    def check_shuffled_batches(batches, order_seed=12345):
        # Get the expected order
        order = np.random.RandomState(order_seed).permutation(45)
        # Three batches
        assert len(batches) == 3
        # Two items in each batch
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        assert len(batches[2]) == 2
        # Verify values
        assert (batches[0][0] == X[order[:15]]).all()
        assert (batches[0][1] == Y[order[:15]]).all()
        assert (batches[1][0] == X[order[15:30]]).all()
        assert (batches[1][1] == Y[order[15:30]]).all()
        assert (batches[2][0] == X[order[30:]]).all()
        assert (batches[2][1] == Y[order[30:]]).all()

    # Function to make iterator factory (callable) that checks its
    # keyword arguments
    def make_iterator_callable(**expected):
        c = make_batch_iterator_callable(X, Y)

        def f(batch_size, **kwargs):
            assert kwargs == expected
            for k in expected.keys():
                del kwargs[k]
            return c(batch_size, **kwargs)
        return f

    cds = data_source.CallableDataSource(make_batch_iterator_callable(X, Y))

    # Not random access
    assert not cds.is_random_access

    # Length
    assert cds.num_samples() is None

    # Three in-order batches
    batches = list(cds.batch_iterator(batch_size=15))
    check_in_order_batches(batches)

    # Three shuffled batches
    batches = list(cds.batch_iterator(
        batch_size=15, shuffle=np.random.RandomState(12345)))
    check_shuffled_batches(batches)

    # Check that keyword args make it over
    cds_2 = data_source.CallableDataSource(
        make_iterator_callable(foo=42, bar=3.14))
    cds_2.batch_iterator(5, foo=42, bar=3.14)

    # Number of samples function
    def num_samples_fn(**kwargs):
        assert kwargs == {'foo': 42, 'bar': 3.14}
        return X.shape[0]

    cds_3 = data_source.CallableDataSource(
        make_iterator_callable(foo=42, bar=3.14),
        num_samples_fn
    )
    assert cds_3.num_samples(foo=42, bar=3.14) == X.shape[0]
    cds_3.batch_iterator(5, foo=42, bar=3.14)

    # Number of samples literal value
    cds_42 = data_source.CallableDataSource(
        make_iterator_callable(foo=42, bar=3.14), 42
    )
    assert cds_42.num_samples(foo=42, bar=3.14) == 42
    cds_42.batch_iterator(5, foo=42, bar=3.14)

    cds_inf = data_source.CallableDataSource(
        make_iterator_callable(foo=42, bar=3.14), np.inf
    )
    assert cds_inf.num_samples(foo=42, bar=3.14) == np.inf
    cds_inf.batch_iterator(5, foo=42, bar=3.14)

    with pytest.raises(TypeError):
        data_source.CallableDataSource(
            make_iterator_callable(foo=42, bar=3.14), 'invalid'
        )


def test_IteratorDataSource():
    from batchup import data_source

    # Data to extract batches from
    X = np.arange(45)
    Y = np.arange(90).reshape((45, 2))

    # Helper functions to check
    def check_in_order_batches(batches):
        # Three batches
        assert len(batches) == 3
        # Two items in each batch
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        assert len(batches[2]) == 2
        # Verify values
        assert (batches[0][0] == X[:15]).all()
        assert (batches[0][1] == Y[:15]).all()
        assert (batches[1][0] == X[15:30]).all()
        assert (batches[1][1] == Y[15:30]).all()
        assert (batches[2][0] == X[30:]).all()
        assert (batches[2][1] == Y[30:]).all()

    def check_shuffled_batches(batches, order_seed=12345):
        # Get the expected order
        order = np.random.RandomState(order_seed).permutation(45)
        # Three batches
        assert len(batches) == 3
        # Two items in each batch
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        assert len(batches[2]) == 2
        # Verify values
        assert (batches[0][0] == X[order[:15]]).all()
        assert (batches[0][1] == Y[order[:15]]).all()
        assert (batches[1][0] == X[order[15:30]]).all()
        assert (batches[1][1] == Y[order[15:30]]).all()
        assert (batches[2][0] == X[order[30:]]).all()
        assert (batches[2][1] == Y[order[30:]]).all()

    # Re-use the function defined above to create the iterator
    in_order_batch_iter = make_batch_iterator_callable(X, Y)(15)
    ds = data_source.IteratorDataSource(in_order_batch_iter)

    # Not random access
    assert not ds.is_random_access

    assert ds.num_samples() is None
    batches = list(ds.batch_iterator(batch_size=15))
    check_in_order_batches(batches)

    # Three shuffled batches
    shuffled_batch_iter = make_batch_iterator_callable(X, Y)(
        15, shuffle=np.random.RandomState(12345))
    ds = data_source.IteratorDataSource(shuffled_batch_iter)
    assert ds.num_samples() is None
    batches = list(ds.batch_iterator(batch_size=15))
    check_shuffled_batches(batches)

    # With number of samples
    ds_42 = data_source.IteratorDataSource(in_order_batch_iter, 42)
    assert ds_42.num_samples() == 42

    ds_inf = data_source.IteratorDataSource(in_order_batch_iter, np.inf)
    assert ds_inf.num_samples() == np.inf

    with pytest.raises(TypeError):
        data_source.IteratorDataSource(in_order_batch_iter, 'invalid')


def test_CompositeDataSource():
    from batchup import data_source

    # Test `CompositeDataSource` using an example layout; a generative
    # adversarial network (GAN) for semi-supervised learning
    # We have:
    # - 15 supervised samples with ground truths; `sup_X`, `sup_y`
    # - 33 unsupervised samples `unsup_X`
    sup_X = np.random.normal(size=(15, 10))
    sup_y = np.random.randint(0, 10, size=(15,))
    unsup_X = np.random.normal(size=(33, 10))

    # We need a dataset containing the supervised samples
    sup_ds = data_source.ArrayDataSource([sup_X, sup_y], repeats=-1)
    # We need a dataset containing the unsupervised samples
    unsup_ds = data_source.ArrayDataSource([unsup_X])

    # We need to:
    # - repeatedly iterate over the supervised samples
    # - iterate over the unsupervised samples for the generator
    # - iterate over the unsupervised samples again in a different order
    #   for the discriminator
    gan_ds = data_source.CompositeDataSource([
        sup_ds, unsup_ds, unsup_ds
    ])
    struct_gan_ds = data_source.CompositeDataSource([
        sup_ds, unsup_ds, unsup_ds
    ], flatten=False)

    # Check number of samples
    assert gan_ds.num_samples() == 33

    def check_structured_batch_layout(batch):
        # Layout is:
        # ((sup_x, sup_y), (gen_x,), (disc_x,))
        assert isinstance(batch, tuple)
        assert isinstance(batch[0], tuple)
        assert isinstance(batch[1], tuple)
        assert isinstance(batch[2], tuple)
        assert len(batch) == 3
        assert len(batch[0]) == 2
        assert len(batch[1]) == 1
        assert len(batch[2]) == 1

    batches = list(gan_ds.batch_iterator(
        batch_size=10, shuffle=np.random.RandomState(12345)))
    # Get the expected order for the supervised, generator and discriminator
    # sets
    # Note: draw in the same order that the data source will
    order_rng = np.random.RandomState(12345)
    order_sup = order_rng.permutation(15)
    order_gen = order_rng.permutation(33)
    order_dis = order_rng.permutation(33)
    order_sup = np.append(order_sup, order_rng.permutation(15))
    order_sup = np.append(order_sup, order_rng.permutation(15))
    order_gen = np.append(order_gen, order_rng.permutation(33))
    order_dis = np.append(order_dis, order_rng.permutation(33))
    # Four batches
    assert len(batches) == 4
    # Four items in each batch
    assert len(batches[0]) == 4
    assert len(batches[1]) == 4
    assert len(batches[2]) == 4
    assert len(batches[3]) == 4
    # Verify values
    assert (batches[0][0] == sup_X[order_sup[:10]]).all()
    assert (batches[0][1] == sup_y[order_sup[:10]]).all()
    assert (batches[0][2] == unsup_X[order_gen[:10]]).all()
    assert (batches[0][3] == unsup_X[order_dis[:10]]).all()

    assert (batches[1][0] == sup_X[order_sup[10:20]]).all()
    assert (batches[1][1] == sup_y[order_sup[10:20]]).all()
    assert (batches[1][2] == unsup_X[order_gen[10:20]]).all()
    assert (batches[1][3] == unsup_X[order_dis[10:20]]).all()

    assert (batches[2][0] == sup_X[order_sup[20:30]]).all()
    assert (batches[2][1] == sup_y[order_sup[20:30]]).all()
    assert (batches[2][2] == unsup_X[order_gen[20:30]]).all()
    assert (batches[2][3] == unsup_X[order_dis[20:30]]).all()

    assert (batches[3][0] == sup_X[order_sup[30:33]]).all()
    assert (batches[3][1] == sup_y[order_sup[30:33]]).all()
    assert (batches[3][2] == unsup_X[order_gen[30:33]]).all()
    assert (batches[3][3] == unsup_X[order_dis[30:33]]).all()

    # Now disable flattening, resulting in structured batches:
    batches = list(struct_gan_ds.batch_iterator(
        batch_size=10, shuffle=np.random.RandomState(12345)))

    # Four batches
    assert len(batches) == 4
    # Two items in each batch
    check_structured_batch_layout(batches[0])
    check_structured_batch_layout(batches[1])
    check_structured_batch_layout(batches[2])
    check_structured_batch_layout(batches[3])
    # Verify values
    assert (batches[0][0][0] == sup_X[order_sup[:10]]).all()
    assert (batches[0][0][1] == sup_y[order_sup[:10]]).all()
    assert (batches[0][1][0] == unsup_X[order_gen[:10]]).all()
    assert (batches[0][2][0] == unsup_X[order_dis[:10]]).all()

    assert (batches[1][0][0] == sup_X[order_sup[10:20]]).all()
    assert (batches[1][0][1] == sup_y[order_sup[10:20]]).all()
    assert (batches[1][1][0] == unsup_X[order_gen[10:20]]).all()
    assert (batches[1][2][0] == unsup_X[order_dis[10:20]]).all()

    assert (batches[2][0][0] == sup_X[order_sup[20:30]]).all()
    assert (batches[2][0][1] == sup_y[order_sup[20:30]]).all()
    assert (batches[2][1][0] == unsup_X[order_gen[20:30]]).all()
    assert (batches[2][2][0] == unsup_X[order_dis[20:30]]).all()

    assert (batches[3][0][0] == sup_X[order_sup[30:33]]).all()
    assert (batches[3][0][1] == sup_y[order_sup[30:33]]).all()
    assert (batches[3][1][0] == unsup_X[order_gen[30:33]]).all()
    assert (batches[3][2][0] == unsup_X[order_dis[30:33]]).all()


def test_CompositeDataSource_in_order():
    from batchup import data_source

    # Test `CompositeDataSource` using an example layout; a generative
    # adversarial network (GAN) for semi-supervised learning
    # We have:
    # - 15 supervised samples with ground truths; `sup_X`, `sup_y`
    # - 33 unsupervised samples `unsup_X`
    sup_X = np.random.normal(size=(15, 10))
    sup_y = np.random.randint(0, 10, size=(15,))
    unsup_X = np.random.normal(size=(33, 10))

    # We need a dataset containing the supervised samples
    sup_ds = data_source.ArrayDataSource([sup_X, sup_y], repeats=-1)
    # We need a dataset containing the unsupervised samples
    unsup_ds = data_source.ArrayDataSource([unsup_X])

    # We need to:
    # - repeatedly iterate over the supervised samples
    # - iterate over the unsupervised samples for the generator
    # - iterate over the unsupervised samples again in a different order
    #   for the discriminator
    gan_ds = data_source.CompositeDataSource([
        sup_ds, unsup_ds, unsup_ds
    ])
    struct_gan_ds = data_source.CompositeDataSource([
        sup_ds, unsup_ds, unsup_ds
    ], flatten=False)

    # Check number of samples
    assert gan_ds.num_samples() == 33

    def check_structured_batch_layout(batch):
        # Layout is:
        # ((sup_x, sup_y), (gen_x,), (disc_x,))
        assert isinstance(batch, tuple)
        assert isinstance(batch[0], tuple)
        assert isinstance(batch[1], tuple)
        assert isinstance(batch[2], tuple)
        assert len(batch) == 3
        assert len(batch[0]) == 2
        assert len(batch[1]) == 1
        assert len(batch[2]) == 1

    batches = list(gan_ds.batch_iterator(batch_size=10))
    # Get the expected order for the supervised, generator and discriminator
    # sets
    order_sup = np.concatenate([np.arange(15)] * 3)
    order_gen = np.concatenate([np.arange(33)] * 2)
    order_dis = np.concatenate([np.arange(33)] * 2)
    # Four batches
    assert len(batches) == 4
    # Four items in each batch
    assert len(batches[0]) == 4
    assert len(batches[1]) == 4
    assert len(batches[2]) == 4
    assert len(batches[3]) == 4
    # Verify values
    assert (batches[0][0] == sup_X[order_sup[:10]]).all()
    assert (batches[0][1] == sup_y[order_sup[:10]]).all()
    assert (batches[0][2] == unsup_X[order_gen[:10]]).all()
    assert (batches[0][3] == unsup_X[order_dis[:10]]).all()

    assert (batches[1][0] == sup_X[order_sup[10:20]]).all()
    assert (batches[1][1] == sup_y[order_sup[10:20]]).all()
    assert (batches[1][2] == unsup_X[order_gen[10:20]]).all()
    assert (batches[1][3] == unsup_X[order_dis[10:20]]).all()

    assert (batches[2][0] == sup_X[order_sup[20:30]]).all()
    assert (batches[2][1] == sup_y[order_sup[20:30]]).all()
    assert (batches[2][2] == unsup_X[order_gen[20:30]]).all()
    assert (batches[2][3] == unsup_X[order_dis[20:30]]).all()

    assert (batches[3][0] == sup_X[order_sup[30:33]]).all()
    assert (batches[3][1] == sup_y[order_sup[30:33]]).all()
    assert (batches[3][2] == unsup_X[order_gen[30:33]]).all()
    assert (batches[3][3] == unsup_X[order_dis[30:33]]).all()

    # Now disable flattening, resulting in structured batches:
    batches = list(struct_gan_ds.batch_iterator(batch_size=10))

    # Four batches
    assert len(batches) == 4
    # Two items in each batch
    check_structured_batch_layout(batches[0])
    check_structured_batch_layout(batches[1])
    check_structured_batch_layout(batches[2])
    check_structured_batch_layout(batches[3])
    # Verify values
    assert (batches[0][0][0] == sup_X[order_sup[:10]]).all()
    assert (batches[0][0][1] == sup_y[order_sup[:10]]).all()
    assert (batches[0][1][0] == unsup_X[order_gen[:10]]).all()
    assert (batches[0][2][0] == unsup_X[order_dis[:10]]).all()

    assert (batches[1][0][0] == sup_X[order_sup[10:20]]).all()
    assert (batches[1][0][1] == sup_y[order_sup[10:20]]).all()
    assert (batches[1][1][0] == unsup_X[order_gen[10:20]]).all()
    assert (batches[1][2][0] == unsup_X[order_dis[10:20]]).all()

    assert (batches[2][0][0] == sup_X[order_sup[20:30]]).all()
    assert (batches[2][0][1] == sup_y[order_sup[20:30]]).all()
    assert (batches[2][1][0] == unsup_X[order_gen[20:30]]).all()
    assert (batches[2][2][0] == unsup_X[order_dis[20:30]]).all()

    assert (batches[3][0][0] == sup_X[order_sup[30:33]]).all()
    assert (batches[3][0][1] == sup_y[order_sup[30:33]]).all()
    assert (batches[3][1][0] == unsup_X[order_gen[30:33]]).all()
    assert (batches[3][2][0] == unsup_X[order_dis[30:33]]).all()


def test_CompositeDataSource_batch_indices_iterator():
    from batchup import data_source

    # Test `CompositeDataSource` using an example layout; a generative
    # adversarial network (GAN) for semi-supervised learning
    # We have:
    # - 15 supervised samples with ground truths; `sup_X`, `sup_y`
    # - 33 unsupervised samples `unsup_X`
    sup_X = np.random.normal(size=(15, 10))
    sup_y = np.random.randint(0, 10, size=(15,))
    unsup_X = np.random.normal(size=(33, 10))

    # We need a dataset containing the supervised samples
    sup_ds = data_source.ArrayDataSource([sup_X, sup_y], repeats=-1)
    # We need a dataset containing the unsupervised samples
    unsup_ds = data_source.ArrayDataSource([unsup_X])

    # We need to:
    # - repeatedly iterate over the supervised samples
    # - iterate over the unsupervised samples for the generator
    # - iterate over the unsupervised samples again in a different order
    #   for the discriminator
    gan_ds = data_source.CompositeDataSource([
        sup_ds, unsup_ds, unsup_ds
    ])

    # Check number of samples
    assert gan_ds.num_samples() == 33

    batches = list(gan_ds.batch_indices_iterator(
        batch_size=10, shuffle=np.random.RandomState(12345)))
    # Get the expected order for the supervised, generator and discriminator
    # sets
    # Note: draw in the same order that the data source will
    order_rng = np.random.RandomState(12345)
    order_sup = order_rng.permutation(15)
    order_gen = order_rng.permutation(33)
    order_dis = order_rng.permutation(33)
    order_sup = np.append(order_sup, order_rng.permutation(15))
    order_sup = np.append(order_sup, order_rng.permutation(15))
    order_gen = np.append(order_gen, order_rng.permutation(33))
    order_dis = np.append(order_dis, order_rng.permutation(33))
    # Four batches
    assert len(batches) == 4
    # Four items in each batch
    assert len(batches[0]) == 3
    assert len(batches[1]) == 3
    assert len(batches[2]) == 3
    assert len(batches[3]) == 3
    # Verify values
    assert (batches[0][0] == order_sup[:10]).all()
    assert (batches[0][1] == order_gen[:10]).all()
    assert (batches[0][2] == order_dis[:10]).all()

    assert (batches[1][0] == order_sup[10:20]).all()
    assert (batches[1][1] == order_gen[10:20]).all()
    assert (batches[1][2] == order_dis[10:20]).all()

    assert (batches[2][0] == order_sup[20:30]).all()
    assert (batches[2][1] == order_gen[20:30]).all()
    assert (batches[2][2] == order_dis[20:30]).all()

    assert (batches[3][0] == order_sup[30:33]).all()
    assert (batches[3][1] == order_gen[30:33]).all()
    assert (batches[3][2] == order_dis[30:33]).all()


def test_CompositeDataSource_batch_indices_iterator_in_order():
    from batchup import data_source

    # Test `CompositeDataSource` using an example layout; a generative
    # adversarial network (GAN) for semi-supervised learning
    # We have:
    # - 15 supervised samples with ground truths; `sup_X`, `sup_y`
    # - 33 unsupervised samples `unsup_X`
    sup_X = np.random.normal(size=(15, 10))
    sup_y = np.random.randint(0, 10, size=(15,))
    unsup_X = np.random.normal(size=(33, 10))

    # We need a dataset containing the supervised samples
    sup_ds = data_source.ArrayDataSource([sup_X, sup_y], repeats=-1)
    # We need a dataset containing the unsupervised samples
    unsup_ds = data_source.ArrayDataSource([unsup_X])

    # We need to:
    # - repeatedly iterate over the supervised samples
    # - iterate over the unsupervised samples for the generator
    # - iterate over the unsupervised samples again in a different order
    #   for the discriminator
    gan_ds = data_source.CompositeDataSource([
        sup_ds, unsup_ds, unsup_ds
    ])

    # Check number of samples
    assert gan_ds.num_samples() == 33

    batches = list(gan_ds.batch_indices_iterator(batch_size=10))
    # Get the expected order for the supervised, generator and discriminator
    # sets
    # Note: draw in the same order that the data source will
    order_sup = np.concatenate([np.arange(15)] * 3, axis=0)
    order_unsup = np.concatenate([np.arange(33)] * 2, axis=0)
    # Four batches
    assert len(batches) == 4
    # Four items in each batch
    assert len(batches[0]) == 3
    assert len(batches[1]) == 3
    assert len(batches[2]) == 3
    assert len(batches[3]) == 3
    # Verify values
    assert (batches[0][0] == order_sup[:10]).all()
    assert (batches[0][1] == order_unsup[:10]).all()
    assert (batches[0][2] == order_unsup[:10]).all()

    assert (batches[1][0] == order_sup[10:20]).all()
    assert (batches[1][1] == order_unsup[10:20]).all()
    assert (batches[1][2] == order_unsup[10:20]).all()

    assert (batches[2][0] == order_sup[20:30]).all()
    assert (batches[2][1] == order_unsup[20:30]).all()
    assert (batches[2][2] == order_unsup[20:30]).all()

    assert (batches[3][0] == order_sup[30:33]).all()
    assert (batches[3][1] == order_unsup[30:33]).all()
    assert (batches[3][2] == order_unsup[30:33]).all()


def test_CompositeDataSource_no_trim():
    from batchup import data_source

    # Test `CompositeDataSource` using an example layout; a generative
    # adversarial network (GAN) for semi-supervised learning
    # We have:
    # - 15 supervised samples with ground truths; `sup_X`, `sup_y`
    # - 33 unsupervised samples `unsup_X`
    sup_X = np.random.normal(size=(15, 10))
    sup_y = np.random.randint(0, 10, size=(15,))
    unsup_X = np.random.normal(size=(33, 10))

    # We need a dataset containing the supervised samples
    sup_ds = data_source.ArrayDataSource([sup_X, sup_y], repeats=-1)
    # We need a dataset containing the unsupervised samples
    unsup_ds = data_source.ArrayDataSource([unsup_X])

    # We need to:
    # - repeatedly iterate over the supervised samples
    # - iterate over the unsupervised samples for the generator
    # - iterate over the unsupervised samples again in a different order
    #   for the discriminator
    gan_ds = data_source.CompositeDataSource([
        sup_ds, unsup_ds, unsup_ds
    ], trim=False)
    struct_gan_ds = data_source.CompositeDataSource([
        sup_ds, unsup_ds, unsup_ds
    ], flatten=False, trim=False)

    # Check number of samples
    assert gan_ds.num_samples() == 33

    def check_structured_batch_layout(batch):
        # Layout is:
        # ((sup_x, sup_y), (gen_x,), (disc_x,))
        assert isinstance(batch, tuple)
        assert isinstance(batch[0], tuple)
        assert isinstance(batch[1], tuple)
        assert isinstance(batch[2], tuple)
        assert len(batch) == 3
        assert len(batch[0]) == 2
        assert len(batch[1]) == 1
        assert len(batch[2]) == 1

    batches = list(gan_ds.batch_iterator(
        batch_size=10, shuffle=np.random.RandomState(12345)))
    # Get the expected order for the supervised, generator and discriminator
    # sets
    # Note: draw in the same order that the data source will
    order_rng = np.random.RandomState(12345)
    order_sup = order_rng.permutation(15)
    order_gen = order_rng.permutation(33)
    order_dis = order_rng.permutation(33)
    order_sup = np.append(order_sup, order_rng.permutation(15))
    order_sup = np.append(order_sup, order_rng.permutation(15))
    order_gen = np.append(order_gen, order_rng.permutation(33))
    order_dis = np.append(order_dis, order_rng.permutation(33))
    # Four batches
    assert len(batches) == 4
    # Four items in each batch
    assert len(batches[0]) == 4
    assert len(batches[1]) == 4
    assert len(batches[2]) == 4
    assert len(batches[3]) == 4
    # Verify values
    assert (batches[0][0] == sup_X[order_sup[:10]]).all()
    assert (batches[0][1] == sup_y[order_sup[:10]]).all()
    assert (batches[0][2] == unsup_X[order_gen[:10]]).all()
    assert (batches[0][3] == unsup_X[order_dis[:10]]).all()

    assert (batches[1][0] == sup_X[order_sup[10:20]]).all()
    assert (batches[1][1] == sup_y[order_sup[10:20]]).all()
    assert (batches[1][2] == unsup_X[order_gen[10:20]]).all()
    assert (batches[1][3] == unsup_X[order_dis[10:20]]).all()

    assert (batches[2][0] == sup_X[order_sup[20:30]]).all()
    assert (batches[2][1] == sup_y[order_sup[20:30]]).all()
    assert (batches[2][2] == unsup_X[order_gen[20:30]]).all()
    assert (batches[2][3] == unsup_X[order_dis[20:30]]).all()

    assert (batches[3][0] == sup_X[order_sup[30:40]]).all()
    assert (batches[3][1] == sup_y[order_sup[30:40]]).all()
    assert (batches[3][2] == unsup_X[order_gen[30:33]]).all()
    assert (batches[3][3] == unsup_X[order_dis[30:33]]).all()

    # Now disable flattening, resulting in structured batches:
    batches = list(struct_gan_ds.batch_iterator(
        batch_size=10, shuffle=np.random.RandomState(12345)))

    # Four batches
    assert len(batches) == 4
    # Two items in each batch
    check_structured_batch_layout(batches[0])
    check_structured_batch_layout(batches[1])
    check_structured_batch_layout(batches[2])
    check_structured_batch_layout(batches[3])
    # Verify values
    assert (batches[0][0][0] == sup_X[order_sup[:10]]).all()
    assert (batches[0][0][1] == sup_y[order_sup[:10]]).all()
    assert (batches[0][1][0] == unsup_X[order_gen[:10]]).all()
    assert (batches[0][2][0] == unsup_X[order_dis[:10]]).all()

    assert (batches[1][0][0] == sup_X[order_sup[10:20]]).all()
    assert (batches[1][0][1] == sup_y[order_sup[10:20]]).all()
    assert (batches[1][1][0] == unsup_X[order_gen[10:20]]).all()
    assert (batches[1][2][0] == unsup_X[order_dis[10:20]]).all()

    assert (batches[2][0][0] == sup_X[order_sup[20:30]]).all()
    assert (batches[2][0][1] == sup_y[order_sup[20:30]]).all()
    assert (batches[2][1][0] == unsup_X[order_gen[20:30]]).all()
    assert (batches[2][2][0] == unsup_X[order_dis[20:30]]).all()

    assert (batches[3][0][0] == sup_X[order_sup[30:40]]).all()
    assert (batches[3][0][1] == sup_y[order_sup[30:40]]).all()
    assert (batches[3][1][0] == unsup_X[order_gen[30:33]]).all()
    assert (batches[3][2][0] == unsup_X[order_dis[30:33]]).all()


def test_CompositeDataSource_random_access():
    from batchup import data_source

    # Test the `samples_by_indices` and `samples_by_indices_no_mapping`
    # methods.

    # Test `CompositeDataSource` using an example layout; a generative
    # adversarial network (GAN) for semi-supervised learning
    # We have:
    # - 15 supervised samples with ground truths; `sup_X`, `sup_y`
    # - 33 unsupervised samples `unsup_X`
    sup_X = np.random.normal(size=(15, 10))
    sup_y = np.random.randint(0, 10, size=(15,))
    unsup_X = np.random.normal(size=(33, 10))

    # We need a dataset containing the supervised samples
    sup_ds = data_source.ArrayDataSource([sup_X, sup_y], repeats=-1)
    # We need a dataset containing the unsupervised samples
    unsup_ds = data_source.ArrayDataSource([unsup_X])

    # We need to:
    # - repeatedly iterate over the supervised samples
    # - iterate over the unsupervised samples for the generator
    # - iterate over the unsupervised samples again in a different order
    #   for the discriminator
    gan_ds = data_source.CompositeDataSource([
        sup_ds, unsup_ds, unsup_ds
    ])
    struct_gan_ds = data_source.CompositeDataSource([
        sup_ds, unsup_ds, unsup_ds
    ], flatten=False)

    # Check number of samples
    assert gan_ds.num_samples() == 33

    def check_structured_batch_layout(batch):
        # Layout is:
        # ((sup_x, sup_y), (gen_x,), (disc_x,))
        assert isinstance(batch, tuple)
        assert isinstance(batch[0], tuple)
        assert isinstance(batch[1], tuple)
        assert isinstance(batch[2], tuple)
        assert len(batch) == 3
        assert len(batch[0]) == 2
        assert len(batch[1]) == 1
        assert len(batch[2]) == 1

    ndx_batches = list(gan_ds.batch_indices_iterator(
        batch_size=10, shuffle=np.random.RandomState(12345)))
    # Get the expected order for the supervised, generator and discriminator
    # sets
    # Note: draw in the same order that the data source will
    order_rng = np.random.RandomState(12345)
    order_sup = order_rng.permutation(15)
    order_gen = order_rng.permutation(33)
    order_dis = order_rng.permutation(33)
    order_sup = np.append(order_sup, order_rng.permutation(15))
    order_sup = np.append(order_sup, order_rng.permutation(15))
    order_gen = np.append(order_gen, order_rng.permutation(33))
    order_dis = np.append(order_dis, order_rng.permutation(33))
    # Four batches
    assert len(ndx_batches) == 4
    # Verify index values
    assert (ndx_batches[0][0] == order_sup[:10]).all()
    assert (ndx_batches[0][1] == order_gen[:10]).all()
    assert (ndx_batches[0][2] == order_dis[:10]).all()

    assert (ndx_batches[1][0] == order_sup[10:20]).all()
    assert (ndx_batches[1][1] == order_gen[10:20]).all()
    assert (ndx_batches[1][2] == order_dis[10:20]).all()

    assert (ndx_batches[2][0] == order_sup[20:30]).all()
    assert (ndx_batches[2][1] == order_gen[20:30]).all()
    assert (ndx_batches[2][2] == order_dis[20:30]).all()

    assert (ndx_batches[3][0] == order_sup[30:33]).all()
    assert (ndx_batches[3][1] == order_gen[30:33]).all()
    assert (ndx_batches[3][2] == order_dis[30:33]).all()

    # Verify data batches
    batches = [gan_ds.samples_by_indices(b) for b in ndx_batches]
    # Four batches
    assert len(batches) == 4
    # Four items in each batch
    assert len(batches[0]) == 4
    assert len(batches[1]) == 4
    assert len(batches[2]) == 4
    assert len(batches[3]) == 4
    # Verify values
    assert (batches[0][0] == sup_X[order_sup[:10]]).all()
    assert (batches[0][1] == sup_y[order_sup[:10]]).all()
    assert (batches[0][2] == unsup_X[order_gen[:10]]).all()
    assert (batches[0][3] == unsup_X[order_dis[:10]]).all()

    assert (batches[1][0] == sup_X[order_sup[10:20]]).all()
    assert (batches[1][1] == sup_y[order_sup[10:20]]).all()
    assert (batches[1][2] == unsup_X[order_gen[10:20]]).all()
    assert (batches[1][3] == unsup_X[order_dis[10:20]]).all()

    assert (batches[2][0] == sup_X[order_sup[20:30]]).all()
    assert (batches[2][1] == sup_y[order_sup[20:30]]).all()
    assert (batches[2][2] == unsup_X[order_gen[20:30]]).all()
    assert (batches[2][3] == unsup_X[order_dis[20:30]]).all()

    assert (batches[3][0] == sup_X[order_sup[30:33]]).all()
    assert (batches[3][1] == sup_y[order_sup[30:33]]).all()
    assert (batches[3][2] == unsup_X[order_gen[30:33]]).all()
    assert (batches[3][3] == unsup_X[order_dis[30:33]]).all()

    # Verify unmapped data batches
    batches = [gan_ds.samples_by_indices_nomapping(b) for b in ndx_batches]
    # Four batches
    assert len(batches) == 4
    # Four items in each batch
    assert len(batches[0]) == 4
    assert len(batches[1]) == 4
    assert len(batches[2]) == 4
    assert len(batches[3]) == 4
    # Verify values
    assert (batches[0][0] == sup_X[order_sup[:10]]).all()
    assert (batches[0][1] == sup_y[order_sup[:10]]).all()
    assert (batches[0][2] == unsup_X[order_gen[:10]]).all()
    assert (batches[0][3] == unsup_X[order_dis[:10]]).all()

    assert (batches[1][0] == sup_X[order_sup[10:20]]).all()
    assert (batches[1][1] == sup_y[order_sup[10:20]]).all()
    assert (batches[1][2] == unsup_X[order_gen[10:20]]).all()
    assert (batches[1][3] == unsup_X[order_dis[10:20]]).all()

    assert (batches[2][0] == sup_X[order_sup[20:30]]).all()
    assert (batches[2][1] == sup_y[order_sup[20:30]]).all()
    assert (batches[2][2] == unsup_X[order_gen[20:30]]).all()
    assert (batches[2][3] == unsup_X[order_dis[20:30]]).all()

    assert (batches[3][0] == sup_X[order_sup[30:33]]).all()
    assert (batches[3][1] == sup_y[order_sup[30:33]]).all()
    assert (batches[3][2] == unsup_X[order_gen[30:33]]).all()
    assert (batches[3][3] == unsup_X[order_dis[30:33]]).all()

    # Check that incorrectly structured index batches raise an error:
    with pytest.raises(ValueError):
        gan_ds.samples_by_indices([np.arange(5)])

    with pytest.raises(ValueError):
        gan_ds.samples_by_indices_nomapping([np.arange(5)])

    # Now disable flattening, resulting in structured batches:
    ndx_batches = list(struct_gan_ds.batch_indices_iterator(
        batch_size=10, shuffle=np.random.RandomState(12345)))

    # Four batches
    assert len(ndx_batches) == 4
    # Verify values
    assert (ndx_batches[0][0] == order_sup[:10]).all()
    assert (ndx_batches[0][1] == order_gen[:10]).all()
    assert (ndx_batches[0][2] == order_dis[:10]).all()

    assert (ndx_batches[1][0] == order_sup[10:20]).all()
    assert (ndx_batches[1][1] == order_gen[10:20]).all()
    assert (ndx_batches[1][2] == order_dis[10:20]).all()

    assert (ndx_batches[2][0] == order_sup[20:30]).all()
    assert (ndx_batches[2][1] == order_gen[20:30]).all()
    assert (ndx_batches[2][2] == order_dis[20:30]).all()

    assert (ndx_batches[3][0] == order_sup[30:33]).all()
    assert (ndx_batches[3][1] == order_gen[30:33]).all()
    assert (ndx_batches[3][2] == order_dis[30:33]).all()

    # Verify data batches
    batches = [struct_gan_ds.samples_by_indices(b) for b in ndx_batches]
    # Four batches
    assert len(batches) == 4
    # Two items in each batch
    check_structured_batch_layout(batches[0])
    check_structured_batch_layout(batches[1])
    check_structured_batch_layout(batches[2])
    check_structured_batch_layout(batches[3])
    # Verify values
    assert (batches[0][0][0] == sup_X[order_sup[:10]]).all()
    assert (batches[0][0][1] == sup_y[order_sup[:10]]).all()
    assert (batches[0][1][0] == unsup_X[order_gen[:10]]).all()
    assert (batches[0][2][0] == unsup_X[order_dis[:10]]).all()

    assert (batches[1][0][0] == sup_X[order_sup[10:20]]).all()
    assert (batches[1][0][1] == sup_y[order_sup[10:20]]).all()
    assert (batches[1][1][0] == unsup_X[order_gen[10:20]]).all()
    assert (batches[1][2][0] == unsup_X[order_dis[10:20]]).all()

    assert (batches[2][0][0] == sup_X[order_sup[20:30]]).all()
    assert (batches[2][0][1] == sup_y[order_sup[20:30]]).all()
    assert (batches[2][1][0] == unsup_X[order_gen[20:30]]).all()
    assert (batches[2][2][0] == unsup_X[order_dis[20:30]]).all()

    assert (batches[3][0][0] == sup_X[order_sup[30:33]]).all()
    assert (batches[3][0][1] == sup_y[order_sup[30:33]]).all()
    assert (batches[3][1][0] == unsup_X[order_gen[30:33]]).all()
    assert (batches[3][2][0] == unsup_X[order_dis[30:33]]).all()

    # Verify data batches
    batches = [struct_gan_ds.samples_by_indices_nomapping(b)
               for b in ndx_batches]
    # Four batches
    assert len(batches) == 4
    # Two items in each batch
    check_structured_batch_layout(batches[0])
    check_structured_batch_layout(batches[1])
    check_structured_batch_layout(batches[2])
    check_structured_batch_layout(batches[3])
    # Verify values
    assert (batches[0][0][0] == sup_X[order_sup[:10]]).all()
    assert (batches[0][0][1] == sup_y[order_sup[:10]]).all()
    assert (batches[0][1][0] == unsup_X[order_gen[:10]]).all()
    assert (batches[0][2][0] == unsup_X[order_dis[:10]]).all()

    assert (batches[1][0][0] == sup_X[order_sup[10:20]]).all()
    assert (batches[1][0][1] == sup_y[order_sup[10:20]]).all()
    assert (batches[1][1][0] == unsup_X[order_gen[10:20]]).all()
    assert (batches[1][2][0] == unsup_X[order_dis[10:20]]).all()

    assert (batches[2][0][0] == sup_X[order_sup[20:30]]).all()
    assert (batches[2][0][1] == sup_y[order_sup[20:30]]).all()
    assert (batches[2][1][0] == unsup_X[order_gen[20:30]]).all()
    assert (batches[2][2][0] == unsup_X[order_dis[20:30]]).all()

    assert (batches[3][0][0] == sup_X[order_sup[30:33]]).all()
    assert (batches[3][0][1] == sup_y[order_sup[30:33]]).all()
    assert (batches[3][1][0] == unsup_X[order_gen[30:33]]).all()
    assert (batches[3][2][0] == unsup_X[order_dis[30:33]]).all()


def test_CompositeDataSource_random_access_subset():
    from batchup import data_source

    # Test `CompositeDataSource` using an example layout; a generative
    # adversarial network (GAN) for semi-supervised learning
    # We have:
    # - 15 supervised samples with ground truths; `sup_X`, `sup_y`
    # - 33 unsupervised samples `unsup_X`
    sup_X = np.random.normal(size=(15, 10))
    sup_y = np.random.randint(0, 10, size=(15,))
    unsup_X = np.random.normal(size=(33, 10))

    #
    sup_indices = np.random.RandomState(12345).permutation(15)[:10]
    unsup_indices = np.random.RandomState(23456).permutation(33)[:23]

    # We need a dataset containing the supervised samples
    sup_ds = data_source.ArrayDataSource([sup_X, sup_y], indices=sup_indices,
                                         repeats=-1)
    # We need a dataset containing the unsupervised samples
    unsup_ds = data_source.ArrayDataSource([unsup_X], indices=unsup_indices)

    # We need to:
    # - repeatedly iterate over the supervised samples
    # - iterate over the unsupervised samples for the generator
    # - iterate over the unsupervised samples again in a different order
    #   for the discriminator
    gan_ds = data_source.CompositeDataSource([
        sup_ds, unsup_ds, unsup_ds
    ])
    struct_gan_ds = data_source.CompositeDataSource([
        sup_ds, unsup_ds, unsup_ds
    ], flatten=False)

    # Check number of samples
    assert gan_ds.num_samples() == 23

    def check_structured_batch_layout(batch):
        # Layout is:
        # ((sup_x, sup_y), (gen_x,), (disc_x,))
        assert isinstance(batch, tuple)
        assert isinstance(batch[0], tuple)
        assert isinstance(batch[1], tuple)
        assert isinstance(batch[2], tuple)
        assert len(batch) == 3
        assert len(batch[0]) == 2
        assert len(batch[1]) == 1
        assert len(batch[2]) == 1

    mapped_ndx_batches = list(gan_ds.batch_indices_iterator(
        batch_size=10, shuffle=np.random.RandomState(12345)))

    # Get the expected order for the supervised, generator and discriminator
    # sets
    # Note: draw in the same order that the data source will
    order_rng = np.random.RandomState(12345)
    order_sup = order_rng.permutation(10)
    order_gen = order_rng.permutation(23)
    order_dis = order_rng.permutation(23)
    order_sup = np.append(order_sup, order_rng.permutation(10))
    order_sup = np.append(order_sup, order_rng.permutation(10))
    order_gen = np.append(order_gen, order_rng.permutation(23))
    order_dis = np.append(order_dis, order_rng.permutation(23))
    # Three batches
    assert len(mapped_ndx_batches) == 3
    # Verify index values
    assert (mapped_ndx_batches[0][0] == sup_indices[order_sup[:10]]).all()
    assert (mapped_ndx_batches[0][1] == unsup_indices[order_gen[:10]]).all()
    assert (mapped_ndx_batches[0][2] == unsup_indices[order_dis[:10]]).all()

    assert (mapped_ndx_batches[1][0] == sup_indices[order_sup[10:20]]).all()
    assert (mapped_ndx_batches[1][1] == unsup_indices[order_gen[10:20]]).all()
    assert (mapped_ndx_batches[1][2] == unsup_indices[order_dis[10:20]]).all()

    assert (mapped_ndx_batches[2][0] == sup_indices[order_sup[20:23]]).all()
    assert (mapped_ndx_batches[2][1] == unsup_indices[order_gen[20:23]]).all()
    assert (mapped_ndx_batches[2][2] == unsup_indices[order_dis[20:23]]).all()

    # Build list of unmapped index batches
    unmapped_ndx_batches = []
    for batch_i in range(3):
        i = batch_i * 10
        j = min(batch_i * 10 + 10, 23)
        unmapped_ndx_batches.append((order_sup[i:j], order_gen[i:j],
                                     order_dis[i:j]))

    # Verify data batches
    batches = [gan_ds.samples_by_indices(b) for b in unmapped_ndx_batches]
    # Three batches
    assert len(batches) == 3
    # Four items in each batch
    assert len(batches[0]) == 4
    assert len(batches[1]) == 4
    assert len(batches[2]) == 4
    # Verify values
    assert (batches[0][0] == sup_X[sup_indices[order_sup[:10]]]).all()
    assert (batches[0][1] == sup_y[sup_indices[order_sup[:10]]]).all()
    assert (batches[0][2] == unsup_X[unsup_indices[order_gen[:10]]]).all()
    assert (batches[0][3] == unsup_X[unsup_indices[order_dis[:10]]]).all()

    assert (batches[1][0] == sup_X[sup_indices[order_sup[10:20]]]).all()
    assert (batches[1][1] == sup_y[sup_indices[order_sup[10:20]]]).all()
    assert (batches[1][2] == unsup_X[unsup_indices[order_gen[10:20]]]).all()
    assert (batches[1][3] == unsup_X[unsup_indices[order_dis[10:20]]]).all()

    assert (batches[2][0] == sup_X[sup_indices[order_sup[20:23]]]).all()
    assert (batches[2][1] == sup_y[sup_indices[order_sup[20:23]]]).all()
    assert (batches[2][2] == unsup_X[unsup_indices[order_gen[20:23]]]).all()
    assert (batches[2][3] == unsup_X[unsup_indices[order_dis[20:23]]]).all()

    # Verify unmapped data batches
    batches = [gan_ds.samples_by_indices_nomapping(b)
               for b in mapped_ndx_batches]
    # Three batches
    assert len(batches) == 3
    # Four items in each batch
    assert len(batches[0]) == 4
    assert len(batches[1]) == 4
    assert len(batches[2]) == 4
    # Verify values
    assert (batches[0][0] == sup_X[sup_indices[order_sup[:10]]]).all()
    assert (batches[0][1] == sup_y[sup_indices[order_sup[:10]]]).all()
    assert (batches[0][2] == unsup_X[unsup_indices[order_gen[:10]]]).all()
    assert (batches[0][3] == unsup_X[unsup_indices[order_dis[:10]]]).all()

    assert (batches[1][0] == sup_X[sup_indices[order_sup[10:20]]]).all()
    assert (batches[1][1] == sup_y[sup_indices[order_sup[10:20]]]).all()
    assert (batches[1][2] == unsup_X[unsup_indices[order_gen[10:20]]]).all()
    assert (batches[1][3] == unsup_X[unsup_indices[order_dis[10:20]]]).all()

    assert (batches[2][0] == sup_X[sup_indices[order_sup[20:23]]]).all()
    assert (batches[2][1] == sup_y[sup_indices[order_sup[20:23]]]).all()
    assert (batches[2][2] == unsup_X[unsup_indices[order_gen[20:23]]]).all()
    assert (batches[2][3] == unsup_X[unsup_indices[order_dis[20:23]]]).all()

    # Check that incorrectly structured index batches raise an error:
    with pytest.raises(ValueError):
        gan_ds.samples_by_indices([np.arange(5)])

    with pytest.raises(ValueError):
        gan_ds.samples_by_indices_nomapping([np.arange(5)])

    # Now disable flattening, resulting in structured batches:
    mapped_ndx_batches = list(struct_gan_ds.batch_indices_iterator(
        batch_size=10, shuffle=np.random.RandomState(12345)))

    # Three batches
    assert len(mapped_ndx_batches) == 3
    # Verify values
    assert (mapped_ndx_batches[0][0] == sup_indices[order_sup[:10]]).all()
    assert (mapped_ndx_batches[0][1] == unsup_indices[order_gen[:10]]).all()
    assert (mapped_ndx_batches[0][2] == unsup_indices[order_dis[:10]]).all()

    assert (mapped_ndx_batches[1][0] == sup_indices[order_sup[10:20]]).all()
    assert (mapped_ndx_batches[1][1] == unsup_indices[order_gen[10:20]]).all()
    assert (mapped_ndx_batches[1][2] == unsup_indices[order_dis[10:20]]).all()

    assert (mapped_ndx_batches[2][0] == sup_indices[order_sup[20:23]]).all()
    assert (mapped_ndx_batches[2][1] == unsup_indices[order_gen[20:23]]).all()
    assert (mapped_ndx_batches[2][2] == unsup_indices[order_dis[20:23]]).all()

    # Verify data batches
    batches = [struct_gan_ds.samples_by_indices(b)
               for b in unmapped_ndx_batches]
    # Three batches
    assert len(batches) == 3
    # Two items in each batch
    check_structured_batch_layout(batches[0])
    check_structured_batch_layout(batches[1])
    check_structured_batch_layout(batches[2])
    # Verify values
    assert (batches[0][0][0] == sup_X[sup_indices[order_sup[:10]]]).all()
    assert (batches[0][0][1] == sup_y[sup_indices[order_sup[:10]]]).all()
    assert (batches[0][1][0] == unsup_X[unsup_indices[order_gen[:10]]]).all()
    assert (batches[0][2][0] == unsup_X[unsup_indices[order_dis[:10]]]).all()

    assert (batches[1][0][0] == sup_X[sup_indices[order_sup[10:20]]]).all()
    assert (batches[1][0][1] == sup_y[sup_indices[order_sup[10:20]]]).all()
    assert (batches[1][1][0] == unsup_X[unsup_indices[order_gen[10:20]]]).all()
    assert (batches[1][2][0] == unsup_X[unsup_indices[order_dis[10:20]]]).all()

    assert (batches[2][0][0] == sup_X[sup_indices[order_sup[20:23]]]).all()
    assert (batches[2][0][1] == sup_y[sup_indices[order_sup[20:23]]]).all()
    assert (batches[2][1][0] == unsup_X[unsup_indices[order_gen[20:23]]]).all()
    assert (batches[2][2][0] == unsup_X[unsup_indices[order_dis[20:23]]]).all()

    # Verify data batches
    batches = [struct_gan_ds.samples_by_indices_nomapping(b)
               for b in mapped_ndx_batches]
    # Three batches
    assert len(batches) == 3
    # Two items in each batch
    check_structured_batch_layout(batches[0])
    check_structured_batch_layout(batches[1])
    check_structured_batch_layout(batches[2])
    # Verify values
    assert (batches[0][0][0] == sup_X[sup_indices[order_sup[:10]]]).all()
    assert (batches[0][0][1] == sup_y[sup_indices[order_sup[:10]]]).all()
    assert (batches[0][1][0] == unsup_X[unsup_indices[order_gen[:10]]]).all()
    assert (batches[0][2][0] == unsup_X[unsup_indices[order_dis[:10]]]).all()

    assert (batches[1][0][0] == sup_X[sup_indices[order_sup[10:20]]]).all()
    assert (batches[1][0][1] == sup_y[sup_indices[order_sup[10:20]]]).all()
    assert (batches[1][1][0] == unsup_X[unsup_indices[order_gen[10:20]]]).all()
    assert (batches[1][2][0] == unsup_X[unsup_indices[order_dis[10:20]]]).all()

    assert (batches[2][0][0] == sup_X[sup_indices[order_sup[20:23]]]).all()
    assert (batches[2][0][1] == sup_y[sup_indices[order_sup[20:23]]]).all()
    assert (batches[2][1][0] == unsup_X[unsup_indices[order_gen[20:23]]]).all()
    assert (batches[2][2][0] == unsup_X[unsup_indices[order_dis[20:23]]]).all()


def test_CompositeDataSource_no_random_access():
    from batchup import data_source

    # Check that a `CompositeDataSource` composed of data sources
    # that are not random access is not random access itself, and that
    # invoking methods that require random access data sources raises
    # `TypeError`

    # Test `CompositeDataSource` using an example layout; a generative
    # adversarial network (GAN) for semi-supervised learning
    # We have:
    # - 15 supervised samples with ground truths; `sup_X`, `sup_y`
    # - 33 unsupervised samples `unsup_X`
    sup_X = np.random.normal(size=(15, 10))
    sup_y = np.random.randint(0, 10, size=(15,))
    unsup_X = np.random.normal(size=(33, 10))

    sup_call = make_batch_iterator_callable(sup_X, sup_y)

    # We need a dataset containing the supervised samples
    sup_ds = data_source.CallableDataSource(sup_call)
    # We need a dataset containing the unsupervised samples
    unsup_ds = data_source.ArrayDataSource([unsup_X])

    # We need to:
    # - repeatedly iterate over the supervised samples
    # - iterate over the unsupervised samples for the generator
    # - iterate over the unsupervised samples again in a different order
    #   for the discriminator
    gan_ds = data_source.CompositeDataSource([
        sup_ds, unsup_ds, unsup_ds
    ])
    struct_gan_ds = data_source.CompositeDataSource([
        sup_ds, unsup_ds, unsup_ds
    ], flatten=False)

    assert not gan_ds.is_random_access
    assert not struct_gan_ds.is_random_access

    with pytest.raises(TypeError):
        next(gan_ds.batch_indices_iterator(batch_size=10))

    with pytest.raises(TypeError):
        next(gan_ds.samples_by_indices(np.arange(10)))

    with pytest.raises(TypeError):
        next(gan_ds.samples_by_indices_nomapping(np.arange(10)))


def test_ChoiceDataSource():
    from batchup import data_source

    a_X = np.random.normal(size=(40, 3))
    b_X = np.random.normal(size=(20, 3))
    c_X = np.random.normal(size=(10, 3))
    d_X = np.random.normal(size=(5, 3))
    a_ds = data_source.ArrayDataSource([a_X])
    b_ds = data_source.ArrayDataSource([b_X])
    c_ds = data_source.ArrayDataSource([c_X])
    d_ds = data_source.ArrayDataSource([d_X])

    ch_ds = data_source.ChoiceDataSource([a_ds, b_ds, c_ds, d_ds])

    assert ch_ds.is_random_access
    assert ch_ds.num_samples() == 75

    a_call_it = make_batch_iterator_callable(a_X)
    b_call_it = make_batch_iterator_callable(b_X)

    a_call = data_source.CallableDataSource(a_call_it)
    b_call = data_source.CallableDataSource(b_call_it)

    ch_call_ds = data_source.ChoiceDataSource([a_call, b_call, c_ds, d_ds])

    assert not ch_call_ds.is_random_access
    assert ch_call_ds.num_samples() == 15

    # Methods that require random access should not work
    with pytest.raises(TypeError):
        ch_call_ds.samples_by_indices((0, np.arange(5)))

    with pytest.raises(TypeError):
        ch_call_ds.samples_by_indices_nomapping((0, np.arange(5)))

    with pytest.raises(TypeError):
        ch_call_ds.batch_indices_iterator(batch_size=5, shuffle=True)


def test_ChoiceDataSource_ds_indices_stratified():
    from batchup import data_source

    # Test `_ds_indices_stratified_in_order` method

    assert data_source.ChoiceDataSource._ds_indices_stratified_in_order(
        [8, 4, 2, 1]).tolist() == [0, 1, 2, 3,
                                   0,
                                   0, 1,
                                   0,
                                   0, 1, 2,
                                   0,
                                   0, 1,
                                   0]

    assert data_source.ChoiceDataSource._ds_indices_stratified_in_order(
        [7, 3, 1]).tolist() == [0, 1, 2,
                                0,
                                0,
                                0, 1,
                                0,
                                0, 1,
                                0]

    rng = np.random.RandomState(12345)

    for i in range(128):
        n_batches = rng.randint(1, 16, size=(6,))
        ds_i = data_source.ChoiceDataSource._ds_indices_stratified_in_order(
            n_batches)
        hist = np.bincount(ds_i)
        assert (hist == n_batches).all()

    # Test `_ds_indices_stratified_shuffled` method

    for i in range(128):
        n_batches = rng.randint(1, 16, size=(6,))
        ds_i = data_source.ChoiceDataSource._ds_indices_stratified_shuffled(
            n_batches, rng)
        hist = np.bincount(ds_i)
        assert (hist == n_batches).all()


def test_ChoiceDataSource_ds_iterator_not_stratified():
    from batchup import data_source

    # Test `_ds_iterator` method
    a_ds = data_source.ArrayDataSource([np.random.normal(size=(40, 3))])
    b_ds = data_source.ArrayDataSource([np.random.normal(size=(20, 3))])
    c_ds = data_source.ArrayDataSource([np.random.normal(size=(10, 3))])
    d_ds = data_source.ArrayDataSource([np.random.normal(size=(5, 3))])

    def make_iters():
        return [iter(range(8)), iter(range(4)), iter(range(2)),
                iter(range(1))]

    # Non-stratified
    ch_ds = data_source.ChoiceDataSource([a_ds, b_ds, c_ds, d_ds])

    # In-order
    batches = list(ch_ds._ds_iterator(5, make_iters(), None))
    assert len(batches) == 7
    assert batches[0] == (0, 0)
    assert batches[1] == (1, 0)
    assert batches[2] == (2, 0)
    assert batches[3] == (3, 0)
    assert batches[4] == (0, 1)
    assert batches[5] == (1, 1)
    assert batches[6] == (2, 1)

    # Shuffled
    shuffle_rng = np.random.RandomState(12345)
    order_rng = np.random.RandomState(12345)

    ds_order = np.append(order_rng.permutation(4), order_rng.permutation(4))

    batches = list(ch_ds._ds_iterator(5, make_iters(), shuffle_rng))

    # One one batch can be extracted from d_ds
    # So one loop round all datasets then a partial loop until we hit d_ds
    # again
    # Find the index at which we hit it for a second time
    stop_at = np.where(ds_order == 3)[0][1]
    expected_batches = []
    for j in range(2):
        for i in range(4):
            expected_batches.append((ds_order[j*4+i], j))

    assert batches == expected_batches[:stop_at]


def test_ChoiceDataSource_ds_iterator_stratified():
    from batchup import data_source

    # Test `_ds_iterator` method
    a_ds = data_source.ArrayDataSource([np.random.normal(size=(40, 3))])
    b_ds = data_source.ArrayDataSource([np.random.normal(size=(20, 3))])
    c_ds = data_source.ArrayDataSource([np.random.normal(size=(10, 3))])
    d_ds = data_source.ArrayDataSource([np.random.normal(size=(5, 3))])

    def make_iters():
        return [iter(range(8)), iter(range(4)), iter(range(2)),
                iter(range(1))]

    # Non-stratified
    ch_ds = data_source.ChoiceDataSource([a_ds, b_ds, c_ds, d_ds],
                                         stratified=True)

    # In-order
    batches = list(ch_ds._ds_iterator(5, make_iters(), None))
    assert batches == [(0, 0), (1, 0), (2, 0), (3, 0),
                       (0, 1),
                       (0, 2), (1, 1),
                       (0, 3),
                       (0, 4), (1, 2), (2, 1),
                       (0, 5),
                       (0, 6), (1, 3),
                       (0, 7)]

    # Shuffled
    shuffle_rng = np.random.RandomState(12345)
    batches = list(ch_ds._ds_iterator(5, make_iters(), shuffle_rng))
    # Convert to list of lists
    arr_batches = np.array([list(x) for x in batches])
    for i in range(len(arr_batches)):
        hist = np.bincount(arr_batches[:i+1, 0], minlength=4)
        assert hist[arr_batches[i, 0]] == arr_batches[i, 1] + 1


def test_ChoiceDataSource_batch_iterator():
    from batchup import data_source

    a_X = np.arange(120).reshape((40, 3))
    b_X = np.arange(60).reshape((20, 3)) * 1000
    c_X = np.arange(30).reshape((10, 3)) * 100000
    d_X = np.arange(15).reshape((5, 3)) * 10000000
    a_ds = data_source.ArrayDataSource([a_X])
    b_ds = data_source.ArrayDataSource([b_X])
    c_ds = data_source.ArrayDataSource([c_X])
    d_ds = data_source.ArrayDataSource([d_X])

    # Not-stratified
    ch_ds = data_source.ChoiceDataSource([a_ds, b_ds, c_ds, d_ds])

    # In order batches
    batches = list(ch_ds.batch_iterator(5))
    assert len(batches) == 7

    assert (batches[0][0] == a_X[:5]).all()
    assert (batches[1][0] == b_X[:5]).all()
    assert (batches[2][0] == c_X[:5]).all()
    assert (batches[3][0] == d_X[:5]).all()
    assert (batches[4][0] == a_X[5:10]).all()
    assert (batches[5][0] == b_X[5:10]).all()
    assert (batches[6][0] == c_X[5:10]).all()

    # Stratified
    ch_ds = data_source.ChoiceDataSource([a_ds, b_ds, c_ds, d_ds],
                                         stratified=True)

    # In order batches
    batches = list(ch_ds.batch_iterator(5))
    assert len(batches) == 15

    assert (batches[0][0] == a_X[:5]).all()
    assert (batches[1][0] == b_X[:5]).all()
    assert (batches[2][0] == c_X[:5]).all()
    assert (batches[3][0] == d_X[:5]).all()
    assert (batches[4][0] == a_X[5:10]).all()
    assert (batches[5][0] == a_X[10:15]).all()
    assert (batches[6][0] == b_X[5:10]).all()
    assert (batches[7][0] == a_X[15:20]).all()
    assert (batches[8][0] == a_X[20:25]).all()
    assert (batches[9][0] == b_X[10:15]).all()
    assert (batches[10][0] == c_X[5:10]).all()
    assert (batches[11][0] == a_X[25:30]).all()
    assert (batches[12][0] == a_X[30:35]).all()
    assert (batches[13][0] == b_X[15:20]).all()
    assert (batches[14][0] == a_X[35:40]).all()


def test_ChoiceDataSource_batch_indices_iterator():
    from batchup import data_source

    a_X = np.arange(120).reshape((40, 3))
    b_X = np.arange(60).reshape((20, 3)) * 1000
    c_X = np.arange(30).reshape((10, 3)) * 100000
    d_X = np.arange(15).reshape((5, 3)) * 10000000
    a_ds = data_source.ArrayDataSource([a_X])
    b_ds = data_source.ArrayDataSource([b_X])
    c_ds = data_source.ArrayDataSource([c_X])
    d_ds = data_source.ArrayDataSource([d_X])

    # Not-stratified
    ch_ds = data_source.ChoiceDataSource([a_ds, b_ds, c_ds, d_ds])

    # In order batches
    batches = list(ch_ds.batch_indices_iterator(5))
    assert len(batches) == 7

    print(batches[0])

    assert batches[0][0] == 0
    assert batches[1][0] == 1
    assert batches[2][0] == 2
    assert batches[3][0] == 3
    assert batches[4][0] == 0
    assert batches[5][0] == 1
    assert batches[6][0] == 2
    assert (batches[0][1] == np.arange(5)).all()
    assert (batches[1][1] == np.arange(5)).all()
    assert (batches[2][1] == np.arange(5)).all()
    assert (batches[3][1] == np.arange(5)).all()
    assert (batches[4][1] == np.arange(5, 10)).all()
    assert (batches[5][1] == np.arange(5, 10)).all()
    assert (batches[6][1] == np.arange(5, 10)).all()

    # Stratified
    ch_ds = data_source.ChoiceDataSource([a_ds, b_ds, c_ds, d_ds],
                                         stratified=True)

    # In order batches
    batches = list(ch_ds.batch_indices_iterator(5))
    assert len(batches) == 15

    assert batches[0][0] == 0
    assert batches[1][0] == 1
    assert batches[2][0] == 2
    assert batches[3][0] == 3
    assert batches[4][0] == 0
    assert batches[5][0] == 0
    assert batches[6][0] == 1
    assert batches[7][0] == 0
    assert batches[8][0] == 0
    assert batches[9][0] == 1
    assert batches[10][0] == 2
    assert batches[11][0] == 0
    assert batches[12][0] == 0
    assert batches[13][0] == 1
    assert batches[14][0] == 0
    assert (batches[0][1] == np.arange(5)).all()
    assert (batches[1][1] == np.arange(5)).all()
    assert (batches[2][1] == np.arange(5)).all()
    assert (batches[3][1] == np.arange(5)).all()
    assert (batches[4][1] == np.arange(5, 10)).all()
    assert (batches[5][1] == np.arange(10, 15)).all()
    assert (batches[6][1] == np.arange(5, 10)).all()
    assert (batches[7][1] == np.arange(15, 20)).all()
    assert (batches[8][1] == np.arange(20, 25)).all()
    assert (batches[9][1] == np.arange(10, 15)).all()
    assert (batches[10][1] == np.arange(5, 10)).all()
    assert (batches[11][1] == np.arange(25, 30)).all()
    assert (batches[12][1] == np.arange(30, 35)).all()
    assert (batches[13][1] == np.arange(15, 20)).all()
    assert (batches[14][1] == np.arange(35, 40)).all()


def test_ChoiceDataSource_samples_by_indices_no_mapping():
    from batchup import data_source

    a_X = np.random.normal(size=(40, 3))
    b_X = np.random.normal(size=(20, 3))
    c_X = np.random.normal(size=(10, 3))
    d_X = np.random.normal(size=(5, 3))
    a_ds = data_source.ArrayDataSource([a_X])
    b_ds = data_source.ArrayDataSource([b_X])
    c_ds = data_source.ArrayDataSource([c_X])
    d_ds = data_source.ArrayDataSource([d_X])

    ch_ds = data_source.ChoiceDataSource([a_ds, b_ds, c_ds, d_ds])

    assert (ch_ds.samples_by_indices_nomapping((0, np.arange(5))) ==
            a_X[:5]).all()
    assert (ch_ds.samples_by_indices_nomapping((1, np.arange(5))) ==
            b_X[:5]).all()
    assert (ch_ds.samples_by_indices_nomapping((2, np.arange(5))) ==
            c_X[:5]).all()
    assert (ch_ds.samples_by_indices_nomapping((3, np.arange(5))) ==
            d_X[:5]).all()

    # Should only accept tuple
    with pytest.raises(TypeError):
        ch_ds.samples_by_indices_nomapping(np.arange(5))

    with pytest.raises(TypeError):
        ch_ds.samples_by_indices_nomapping([0, np.arange(5)])


def test_ChoiceDataSource_samples_by_indices():
    from batchup import data_source

    a_X = np.random.normal(size=(120, 3))
    b_X = np.random.normal(size=(120, 3))
    c_X = np.random.normal(size=(120, 3))
    d_X = np.random.normal(size=(120, 3))
    a_inds = np.random.permutation(120)[:40]
    b_inds = np.random.permutation(120)[:20]
    c_inds = np.random.permutation(120)[:10]
    d_inds = np.random.permutation(120)[:5]
    a_ds = data_source.ArrayDataSource([a_X], indices=a_inds)
    b_ds = data_source.ArrayDataSource([b_X], indices=b_inds)
    c_ds = data_source.ArrayDataSource([c_X], indices=c_inds)
    d_ds = data_source.ArrayDataSource([d_X], indices=d_inds)

    ch_ds = data_source.ChoiceDataSource([a_ds, b_ds, c_ds, d_ds])

    assert (ch_ds.samples_by_indices_nomapping((0, np.arange(5))) ==
            a_X[:5]).all()
    assert (ch_ds.samples_by_indices_nomapping((1, np.arange(5))) ==
            b_X[:5]).all()
    assert (ch_ds.samples_by_indices_nomapping((2, np.arange(5))) ==
            c_X[:5]).all()
    assert (ch_ds.samples_by_indices_nomapping((3, np.arange(5))) ==
            d_X[:5]).all()

    assert (ch_ds.samples_by_indices((0, np.arange(5))) ==
            a_X[a_inds[:5]]).all()
    assert (ch_ds.samples_by_indices((1, np.arange(5))) ==
            b_X[b_inds[:5]]).all()
    assert (ch_ds.samples_by_indices((2, np.arange(5))) ==
            c_X[c_inds[:5]]).all()
    assert (ch_ds.samples_by_indices((3, np.arange(5))) ==
            d_X[d_inds[:5]]).all()

    # Should only accept tuple
    with pytest.raises(TypeError):
        ch_ds.samples_by_indices(np.arange(5))

    with pytest.raises(TypeError):
        ch_ds.samples_by_indices([0, np.arange(5)])


def test_MapDataSource():
    from batchup import data_source

    # Build some arrays and construct an `ArrayDataSource` for working with
    # them
    X = np.arange(90)
    Y = np.arange(180).reshape((90, 2))
    indices = np.random.RandomState(12345).permutation(90)[:45]
    ds = data_source.ArrayDataSource([X, Y], indices=indices)

    # Define a function that augments the batches with values that are double
    # the input
    def augment(batch_X, batch_Y):
        return [batch_X, batch_Y, batch_X * 2, batch_Y * 2]

    # Build map data source using `map` method
    mds = ds.map(augment)

    # Check the type and contents
    assert isinstance(mds, data_source.MapDataSource)
    assert mds.source is ds
    assert mds.fn is augment

    # Test `num_samples` method
    assert mds.num_samples() == 45

    # Ensure that it is random access as the underlying data source (`ds`) is
    assert mds.is_random_access

    # Test random access methods
    batch = mds.samples_by_indices_nomapping(np.arange(5))
    assert (batch[0] == X[:5]).all()
    assert (batch[1] == Y[:5]).all()
    assert (batch[2] == X[:5] * 2).all()
    assert (batch[3] == Y[:5] * 2).all()

    batch = mds.samples_by_indices(np.arange(5))
    assert (batch[0] == X[indices[:5]]).all()
    assert (batch[1] == Y[indices[:5]]).all()
    assert (batch[2] == X[indices[:5]] * 2).all()
    assert (batch[3] == Y[indices[:5]] * 2).all()

    # Test `batch_iterator`

    # Three in-order batches
    batches = list(mds.batch_iterator(batch_size=15))
    # Three batches
    assert len(batches) == 3
    # Four items in each batch
    assert len(batches[0]) == 4
    assert len(batches[1]) == 4
    assert len(batches[2]) == 4
    # Verify values
    assert (batches[0][0] == X[indices[:15]]).all()
    assert (batches[0][1] == Y[indices[:15]]).all()
    assert (batches[0][2] == X[indices[:15]] * 2).all()
    assert (batches[0][3] == Y[indices[:15]] * 2).all()
    assert (batches[1][0] == X[indices[15:30]]).all()
    assert (batches[1][1] == Y[indices[15:30]]).all()
    assert (batches[1][2] == X[indices[15:30]] * 2).all()
    assert (batches[1][3] == Y[indices[15:30]] * 2).all()
    assert (batches[2][0] == X[indices[30:]]).all()
    assert (batches[2][1] == Y[indices[30:]]).all()
    assert (batches[2][2] == X[indices[30:]] * 2).all()
    assert (batches[2][3] == Y[indices[30:]] * 2).all()

    # Three shuffled batches
    batches = list(mds.batch_iterator(
        batch_size=15, shuffle=np.random.RandomState(12345)))
    # Get the expected order
    order = np.random.RandomState(12345).permutation(45)
    # Three batches
    assert len(batches) == 3
    # Four items in each batch
    assert len(batches[0]) == 4
    assert len(batches[1]) == 4
    assert len(batches[2]) == 4
    # Verify values
    assert (batches[0][0] == X[indices[order[:15]]]).all()
    assert (batches[0][1] == Y[indices[order[:15]]]).all()
    assert (batches[0][2] == X[indices[order[:15]]] * 2).all()
    assert (batches[0][3] == Y[indices[order[:15]]] * 2).all()
    assert (batches[1][0] == X[indices[order[15:30]]]).all()
    assert (batches[1][1] == Y[indices[order[15:30]]]).all()
    assert (batches[1][2] == X[indices[order[15:30]]] * 2).all()
    assert (batches[1][3] == Y[indices[order[15:30]]] * 2).all()
    assert (batches[2][0] == X[indices[order[30:]]]).all()
    assert (batches[2][1] == Y[indices[order[30:]]]).all()
    assert (batches[2][2] == X[indices[order[30:]]] * 2).all()
    assert (batches[2][3] == Y[indices[order[30:]]] * 2).all()

    # Test `batch_indices_iterator`
    ndx_batches = list(mds.batch_indices_iterator(batch_size=15))
    # Three batches
    assert len(ndx_batches) == 3
    # Verify values
    assert (ndx_batches[0] == indices[:15]).all()
    assert (ndx_batches[1] == indices[15:30]).all()
    assert (ndx_batches[2] == indices[30:]).all()

    # Check non-random access data source
    bic = make_batch_iterator_callable(X, Y)
    call_ds = data_source.CallableDataSource(bic)
    m_call_ds = call_ds.map(augment)

    assert not m_call_ds.is_random_access

    with pytest.raises(TypeError):
        next(m_call_ds.batch_indices_iterator(batch_size=10))

    with pytest.raises(TypeError):
        next(m_call_ds.samples_by_indices(np.arange(10)))

    with pytest.raises(TypeError):
        next(m_call_ds.samples_by_indices_nomapping(np.arange(10)))


def test_batch_map_concat():
    from batchup import data_source

    def sqr_sum(x):
        # Ensure that we receive batches of the expected size:
        assert x.shape[0] == 5
        return (x ** 2).sum(axis=1)

    # Construct data to process and create a data source:
    X = np.random.normal(size=(100, 10))
    ds = data_source.ArrayDataSource([X])

    # Apply the function defined above:
    batch_iter = ds.batch_iterator(batch_size=5)
    X_sqr_sum = data_source.batch_map_concat(sqr_sum, batch_iter)
    assert (X_sqr_sum[0] == (X ** 2).sum(axis=1)).all()

    # Process 2 batches at a time:
    batch_iter = ds.batch_iterator(batch_size=5)
    for i in range(5):
        partial_result = data_source.batch_map_concat(sqr_sum, batch_iter,
                                                      n_batches=2)
        # Should have 10 samples per partial result
        assert partial_result[0].shape[0] == 10
        j = i * 10
        assert (partial_result[0] == (X[j:j + 10]**2).sum(axis=1)).all()

    #
    # Multiple return values
    #
    def batch_func(batch_X, batch_Y):
        return (batch_X + 2, (batch_Y**2).sum(axis=1))

    # Dummy progress function to check parameter values
    def progress_iter_func(iterator, total, leave):
        # 47 samples divided into batches of 5 means 10 batches
        assert total == 9
        # not leave
        assert not leave
        return iterator

    # Test `batch_iterator`
    X = np.arange(45)
    Y = np.arange(90).reshape((45, 2))
    ads = data_source.ArrayDataSource([X, Y])

    x, y = data_source.batch_map_concat(batch_func, ads.batch_iterator(5),
                                        progress_iter_func, n_batches=9)

    assert (x == X + 2).all()
    assert (y == (Y**2).sum(axis=1)).all()

    x, y = data_source.batch_map_concat(batch_func, ads.batch_iterator(5),
                                        progress_iter_func, n_batches=9)

    assert (x == X + 2).all()
    assert (y == (Y**2).sum(axis=1)).all()

    # Test prepend_args
    ads_y = data_source.ArrayDataSource([Y])
    x, y = data_source.batch_map_concat(batch_func, ads_y.batch_iterator(5),
                                        progress_iter_func, n_batches=9,
                                        prepend_args=(np.array([5]),))

    assert (x == 5 + 2).all()
    assert (y == (Y**2).sum(axis=1)).all()


def test_batch_map_mean():
    from batchup import data_source

    # Define a function to compute the per-sample binary cross entropy
    # loss:
    def binary_crossentropy_loss(pred, target):
        e = -target * np.log(pred) - (1 - target) * np.log(1 - pred)
        return e.mean(axis=1)

    # Now define a function that computes the *SUM* of the binary cross
    # entropy losses over the sample axis (axis 0), as the default
    # behaviour of `mean_batch_map` will sum them up and divide by the
    # number of samples at the end:
    def binary_crossentropy_loss_sum(pred, target):
        return binary_crossentropy_loss(pred, target).sum()

    # Construct prediction and target data
    pred = np.random.uniform(0.1, 0.9, size=(7, 10))
    tgt = np.random.uniform(0.1, 0.9, size=(7, 10))
    ds = data_source.ArrayDataSource([pred, tgt])

    # Dummy progress function to check parameter values
    def progress_iter_func(iterator, total, leave):
        # 7 samples divided into batches of 5 means 2 batches
        assert total == 2
        # not leave
        assert not leave
        return iterator

    # Apply the loss sum function defined above:
    batch_iter = ds.batch_iterator(batch_size=5)
    loss = data_source.batch_map_mean(binary_crossentropy_loss_sum,
                                      batch_iter, n_batches=2,
                                      progress_iter_func=progress_iter_func)
    assert np.allclose(
        loss, binary_crossentropy_loss(pred, tgt).mean())

    # Have `mean_batch_map` sum over axis 0:
    batch_iter = ds.batch_iterator(batch_size=5)
    loss = data_source.batch_map_mean(binary_crossentropy_loss, batch_iter,
                                      sum_axis=0)
    assert np.allclose(
        loss, binary_crossentropy_loss(pred, tgt).mean())

    # Construct a large data set and use `n_batches` to limit the
    # number of batches processed in one go
    pred_large = np.random.uniform(0.1, 0.9, size=(100, 10))
    tgt_large = np.random.uniform(0.1, 0.9, size=(100, 10))
    ds_large = data_source.ArrayDataSource([pred_large, tgt_large])
    iter_large = ds_large.batch_iterator(batch_size=5)
    for i in range(10):
        partial_loss = data_source.batch_map_mean(
            binary_crossentropy_loss_sum, iter_large, n_batches=2)
        j = i * 10
        assert np.allclose(
            partial_loss, binary_crossentropy_loss(
                pred_large[j:j + 10], tgt_large[j:j + 10]).mean())


def test_data_source_method_batch_map_concat():
    from batchup import data_source

    #
    # Multiple return values
    #
    def batch_func(batch_X, batch_Y):
        return (batch_X + 2, (batch_Y**2).sum(axis=1))

    # Dummy progress function to check parameter values
    def progress_iter_func(iterator, total, leave):
        # 47 samples divided into batches of 5 means 10 batches
        assert total == 9
        # not leave
        assert not leave
        return iterator

    # Test `batch_iterator`
    X = np.arange(45)
    Y = np.arange(90).reshape((45, 2))
    ads = data_source.ArrayDataSource([X, Y])

    x, y = ads.batch_map_concat(batch_func, 5, progress_iter_func,
                                n_batches=9)

    assert (x == X + 2).all()
    assert (y == (Y**2).sum(axis=1)).all()

    x, y = ads.batch_map_concat(batch_func, 5, progress_iter_func)

    assert (x == X + 2).all()
    assert (y == (Y**2).sum(axis=1)).all()

    # Test prepend_args
    ads_y = data_source.ArrayDataSource([Y])
    x, y = ads_y.batch_map_concat(batch_func, 5, progress_iter_func,
                                  prepend_args=(np.array([5]),))

    assert (x == 5 + 2).all()
    assert (y == (Y**2).sum(axis=1)).all()

    # Test batch function with no return value
    def batch_func_no_ret(batch_X, batch_Y):
        pass

    r = ads.batch_map_concat(batch_func_no_ret, 5, progress_iter_func)
    assert r is None

    # Test batch function with single array return value
    def batch_func_one_ret(batch_X, batch_Y):
        return (batch_Y**2).sum(axis=1)

    (y,) = ads.batch_map_concat(batch_func_one_ret, 5, progress_iter_func)

    assert (y == (Y**2).sum(axis=1)).all()

    # Test batch function that returns invalid type
    def batch_func_invalid_ret_type(batch_X, batch_Y):
        return 'invalid'

    with pytest.raises(TypeError):
        ads.batch_map_concat(batch_func_invalid_ret_type, 5,
                             progress_iter_func)

    # Check that using `repeats=-1` without specifying the number of
    # batches raises `ValueError`, as this results in a data source with
    # an infinite number of samples
    ads_inf = data_source.ArrayDataSource([X, Y], repeats=-1)
    with pytest.raises(ValueError):
        ads_inf.batch_map_concat(batch_func, 5, progress_iter_func)

    # Check that using `repeats=-1` while specifying the number of
    # batches is OK. Don't use progress_iter_func as it expects 9 batches,
    # not 15.
    x, y = ads_inf.batch_map_concat(batch_func, 5, n_batches=15)

    assert (x == np.append(X, X[:30], axis=0) + 2).all()
    assert (y == (np.append(Y, Y[:30], axis=0)**2).sum(axis=1)).all()


def test_data_source_method_batch_map_mean_in_order():
    from batchup import data_source

    # Data to extract batches from
    rng = np.random.RandomState(12345)
    X = rng.normal(size=(47,))
    Y = rng.normal(size=(47, 2))
    ads = data_source.ArrayDataSource([X, Y])

    #
    # Multiple return values
    #
    def batch_func(batch_X, batch_Y):
        return (batch_X.sum(), (batch_Y**2).sum(axis=1).sum())

    # Dummy progress function to check parameter values
    def progress_iter_func(iterator, total, leave):
        # 47 samples divided into batches of 5 means 10 batches
        assert total == 10
        # not leave
        assert not leave
        return iterator

    x, y = data_source.batch_map_mean(
        batch_func, ads.batch_iterator(5),
        progress_iter_func=progress_iter_func, sum_axis=None,
        n_batches=10)

    assert np.allclose(x, X.mean())
    assert np.allclose(y, (Y**2).sum(axis=1).mean())

    x, y = ads.batch_map_mean(
        batch_func, 5, progress_iter_func=progress_iter_func, sum_axis=None)

    assert np.allclose(x, X.mean())
    assert np.allclose(y, (Y**2).sum(axis=1).mean())

    #
    # Single return value
    #
    def batch_func_single(batch_X, batch_Y):
        return batch_X.sum()

    # Dummy progress function to check parameter values
    def progress_iter_func(iterator, total, leave):
        # 47 samples divided into batches of 5 means 10 batches
        assert total == 10
        # not leave
        assert not leave
        return iterator

    x, = ads.batch_map_mean(
        batch_func_single, 5, progress_iter_func=progress_iter_func,
        sum_axis=None)

    assert np.allclose(x, X.mean())

    #
    # Batch function that returns no results
    #
    def batch_func_no_results(batch_X, batch_Y):
        return None

    res = ads.batch_map_mean(
        batch_func_no_results, 5, progress_iter_func=progress_iter_func,
        sum_axis=None)

    assert res is None

    #
    # Invalid return value
    #
    def batch_func_invalid(batch_X, batch_Y):
        return 'Should not return a string'

    with pytest.raises(TypeError):
        ads.batch_map_mean(batch_func_invalid, 5)

    #
    # Prepend arguments to batch function
    #
    def batch_func_prepend(a, b, batch_X, batch_Y):
        assert a == 42
        assert b == 3.14
        return (batch_X.sum(), (batch_Y**2).sum(axis=1).sum())

    x, y = ads.batch_map_mean(
        batch_func_prepend, 5, progress_iter_func=progress_iter_func,
        sum_axis=None, prepend_args=(42, 3.14))

    assert np.allclose(x, X.mean())
    assert np.allclose(y, (Y**2).sum(axis=1).mean())

    # Check that using `repeats=-1` without specifying the number of
    # batches raises `ValueError`, as this results in a data source with
    # an infinite number of samples
    ads_inf = data_source.ArrayDataSource([X, Y], repeats=-1)
    with pytest.raises(ValueError):
        ads_inf.batch_map_mean(batch_func, 5, progress_iter_func)

    # Check that using `repeats=-1` while specifying the number of
    # batches is OK. Don't use progress_iter_func as it expects 9 batches,
    # not 15.
    x, y = ads_inf.batch_map_mean(batch_func, 5, n_batches=15)

    assert np.allclose(x, np.append(X, X[:28], axis=0).mean())
    assert np.allclose(
        y, (np.append(Y, Y[:28], axis=0)**2).sum(axis=1).mean())

    # Test a typical loss scenario
    def binary_crossentropy(pred, target):
        e = -target * np.log(pred) - (1 - target) * np.log(1 - pred)
        return e.mean(axis=1).sum(axis=0)

    pred = np.random.uniform(0.0, 1.0, size=(15, 10))
    tgt = np.random.uniform(0.0, 1.0, size=(15, 10))
    ds = data_source.ArrayDataSource([pred, tgt])

    loss = ds.batch_map_mean(binary_crossentropy, batch_size=5)
    assert np.allclose(loss, binary_crossentropy(pred, tgt) / pred.shape[0])

    loss = ds.batch_map_mean(binary_crossentropy, batch_size=5,
                             n_batches=1)
    assert np.allclose(loss, binary_crossentropy(pred[:5], tgt[:5]) / 5.0)


def test_data_source_method_batch_map_mean_in_order_per_sample_func():
    # Test `mean_batch_map` where the batch function returns per-sample
    # results
    from batchup import data_source

    # Data to extract batches from
    rng = np.random.RandomState(12345)
    X = rng.normal(size=(47,))
    Y = rng.normal(size=(47, 2))
    ads = data_source.ArrayDataSource([X, Y])

    #
    # Multiple return values
    #
    def batch_func(batch_X, batch_Y):
        return (batch_X + 2, (batch_Y**2).sum(axis=1))

    # Dummy progress function to check parameter values
    def progress_iter_func(iterator, total, leave):
        # 47 samples divided into batches of 5 means 10 batches
        assert total == 10
        # not leave
        assert not leave
        return iterator

    x, y = ads.batch_map_mean(batch_func, 5,
                              progress_iter_func=progress_iter_func,
                              sum_axis=0)

    assert np.allclose(x, X.mean() + 2.0)
    assert np.allclose(y, (Y**2).sum(axis=1).mean())

    #
    # Single return value
    #
    def batch_func_single(batch_X, batch_Y):
        return batch_X + 2

    (x,) = ads.batch_map_mean(batch_func_single, 5, sum_axis=0)

    assert np.allclose(x, X.mean() + 2.0)
    assert np.allclose(y, (Y**2).sum(axis=1).mean())

    #
    # Batch function that returns no results
    #
    def batch_func_no_results(batch_X, batch_Y):
        return None

    res = ads.batch_map_mean(batch_func_no_results, 5,
                             progress_iter_func=progress_iter_func,
                             sum_axis=0)

    assert res is None


def test_coerce_data_source():
    from batchup import data_source

    rng = np.random.RandomState(12345)
    X = rng.normal(size=(47,))
    Y = rng.normal(size=(47, 2))
    iter_callable = make_batch_iterator_callable(X, Y)

    # Data source objects should be left as is
    ads = data_source.ArrayDataSource([X, Y])
    assert data_source.coerce_data_source(ads) is ads

    # Lists of array-likes should be wrapped in ArrayDataSource
    c_ads = data_source.coerce_data_source([X, Y])
    assert isinstance(c_ads, data_source.ArrayDataSource)
    assert c_ads.data == [X, Y]

    # Callables should be wrapped in CallableDataSource
    c_call = data_source.coerce_data_source(iter_callable)
    assert isinstance(c_call, data_source.CallableDataSource)
    assert c_call.batch_iterator_fn is iter_callable

    # Iterators should be wrapped in IteratorDataSource
    iterator = iter_callable(batch_size=10)
    c_iter = data_source.coerce_data_source(iterator)
    assert isinstance(c_iter, data_source.IteratorDataSource)
    assert c_iter.batch_iter is iterator

    # Unrecognised type should raise TypeError
    with pytest.raises(TypeError):
        data_source.coerce_data_source(1)

    # Empty sequences should raise ValueError
    with pytest.raises(ValueError):
        data_source.coerce_data_source([])

    with pytest.raises(ValueError):
        data_source.coerce_data_source(())

    # Lists of non-array-likes should also raise TypeError
    with pytest.raises(TypeError):
        data_source.coerce_data_source([1])

    with pytest.raises(TypeError):
        data_source.coerce_data_source([X, 1])

    with pytest.raises(TypeError):
        data_source.coerce_data_source([X, c_ads])
