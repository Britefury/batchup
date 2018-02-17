import pytest
import numpy as np


def test_num_batches():
    from batchup import sampling

    assert sampling.num_batches(20, 5) == 4
    assert sampling.num_batches(21, 5) == 5


def test_AbstractSampler():
    from batchup import sampling

    spl = sampling.AbstractSampler()

    with pytest.raises(NotImplementedError):
        _ = spl.num_indices_generated()

    with pytest.raises(NotImplementedError):
        spl.in_order_indices_batch_iterator(batch_size=5)

    with pytest.raises(NotImplementedError):
        spl.shuffled_indices_batch_iterator(
            batch_size=5, shuffle_rng=np.random.RandomState(12345))


def test_StandardSampler():
    from batchup import sampling

    spl = sampling.StandardSampler(length=10)

    # Test index mapping
    indices_to_map = np.random.randint(0, 10, size=(20,))
    assert (spl.map_indices(indices_to_map) == indices_to_map).all()

    # Test index generation

    # In order
    ndx_iter = spl.in_order_indices_batch_iterator(batch_size=5)
    all = np.arange(10)
    batches = list(ndx_iter)
    assert len(batches) == 2
    assert (all[batches[0]] == np.arange(0, 5)).all()
    assert (all[batches[1]] == np.arange(5, 10)).all()

    # In order, last batch short
    ndx_iter = spl.in_order_indices_batch_iterator(batch_size=6)
    all = np.arange(10)
    batches = list(ndx_iter)
    assert len(batches) == 2
    assert (all[batches[0]] == np.arange(0, 6)).all()
    assert (all[batches[1]] == np.arange(6, 10)).all()

    # Shuffled
    shuffled_ndx_iter = spl.shuffled_indices_batch_iterator(
        batch_size=5, shuffle_rng=np.random.RandomState(12345))
    batches = list(shuffled_ndx_iter)
    order = np.random.RandomState(12345).permutation(10)
    assert len(batches) == 2
    assert (all[batches[0]] == order[0:5]).all()
    assert (all[batches[1]] == order[5:10]).all()

    # Shuffled, last batch short
    shuffled_ndx_iter = spl.shuffled_indices_batch_iterator(
        batch_size=6, shuffle_rng=np.random.RandomState(12345))
    batches = list(shuffled_ndx_iter)
    order = np.random.RandomState(12345).permutation(10)
    assert len(batches) == 2
    assert (all[batches[0]] == order[0:6]).all()
    assert (all[batches[1]] == order[6:10]).all()


def test_StandardSampler_repeated():
    from batchup import sampling

    spl = sampling.StandardSampler(length=10, repeats=3)

    # Test index generation

    # In order
    ndx_iter = spl.in_order_indices_batch_iterator(batch_size=5)
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
    ndx_iter = spl.in_order_indices_batch_iterator(batch_size=7)
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
    shuffled_ndx_iter = spl.shuffled_indices_batch_iterator(
        batch_size=5, shuffle_rng=np.random.RandomState(12345))
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
    shuffled_ndx_iter = spl.shuffled_indices_batch_iterator(
        batch_size=7, shuffle_rng=np.random.RandomState(12345))
    batches = list(shuffled_ndx_iter)
    order_rng = np.random.RandomState(12345)
    order = np.concatenate([order_rng.permutation(10) for _ in range(3)])
    assert len(batches) == 5
    assert (all[batches[0]] == order[0:7]).all()
    assert (all[batches[1]] == order[7:14]).all()
    assert (all[batches[2]] == order[14:21]).all()
    assert (all[batches[3]] == order[21:28]).all()
    assert (all[batches[4]] == order[28:30]).all()

    # Check invalid values for repeats
    with pytest.raises(ValueError):
        _ = sampling.StandardSampler(length=10, repeats=-5)


def test_StandardSampler_repeated_small_dataset():
    from batchup import sampling

    spl_inf = sampling.StandardSampler(length=20, repeats=-1)
    assert spl_inf.num_indices_generated() == np.inf

    # Test index generation

    # In order
    ndx_iter = spl_inf.in_order_indices_batch_iterator(batch_size=64)
    all = np.arange(20)
    batches = [next(ndx_iter) for _ in range(10)]
    assert len(batches) == 10
    x = np.concatenate([all[b] for b in batches], axis=0)
    assert (x == np.tile(np.arange(20), [32])).all()

    # Shuffled
    shuffled_ndx_iter = spl_inf.shuffled_indices_batch_iterator(
        batch_size=64, shuffle_rng=np.random.RandomState(12345))
    batches = [next(shuffled_ndx_iter) for _ in range(10)]
    order_rng = np.random.RandomState(12345)
    order = np.concatenate([order_rng.permutation(20) for _ in range(32)])
    assert len(batches) == 10
    x = np.concatenate([all[b] for b in batches], axis=0)
    assert (x == order).all()

    # Test fixed number of repetitions
    spl_16 = sampling.StandardSampler(length=20, repeats=16)
    assert spl_16.num_indices_generated() == 320

    # In order
    ndx_iter = spl_16.in_order_indices_batch_iterator(batch_size=64)
    all = np.arange(20)
    batches = list(ndx_iter)
    assert len(batches) == 5
    x = np.concatenate([all[b] for b in batches], axis=0)
    assert (x == np.tile(np.arange(20), [16])).all()

    # Shuffled
    shuffled_ndx_iter = spl_16.shuffled_indices_batch_iterator(
        batch_size=64, shuffle_rng=np.random.RandomState(12345))
    batches = list(shuffled_ndx_iter)
    order_rng = np.random.RandomState(12345)
    order = np.concatenate([order_rng.permutation(20) for _ in range(16)])
    assert len(batches) == 5
    x = np.concatenate([all[b] for b in batches], axis=0)
    assert (x == order).all()


def test_SubsetSampler():
    from batchup import sampling

    indices = np.random.RandomState(12345).permutation(20)[:10]
    spl = sampling.SubsetSampler(indices=indices)

    # Test index mapping
    indices_to_map = np.random.randint(0, 10, size=(20,))
    assert (spl.map_indices(indices_to_map) == indices[indices_to_map]).all()

    # Test index generation

    # In order
    ndx_iter = spl.in_order_indices_batch_iterator(batch_size=5)
    all = np.arange(20)
    batches = list(ndx_iter)
    assert len(batches) == 2
    assert (all[batches[0]] == indices[0:5]).all()
    assert (all[batches[1]] == indices[5:10]).all()

    # In order, last batch short
    ndx_iter = spl.in_order_indices_batch_iterator(batch_size=6)
    all = np.arange(20)
    batches = list(ndx_iter)
    assert len(batches) == 2
    assert (all[batches[0]] == indices[0:6]).all()
    assert (all[batches[1]] == indices[6:10]).all()

    # Shuffled
    shuffled_ndx_iter = spl.shuffled_indices_batch_iterator(
        batch_size=5, shuffle_rng=np.random.RandomState(12345))
    batches = list(shuffled_ndx_iter)
    order = np.random.RandomState(12345).permutation(10)
    assert len(batches) == 2
    assert (all[batches[0]] == indices[order[0:5]]).all()
    assert (all[batches[1]] == indices[order[5:10]]).all()

    # Shuffled, last batch short
    shuffled_ndx_iter = spl.shuffled_indices_batch_iterator(
        batch_size=6, shuffle_rng=np.random.RandomState(12345))
    batches = list(shuffled_ndx_iter)
    order = np.random.RandomState(12345).permutation(10)
    assert len(batches) == 2
    assert (all[batches[0]] == indices[order[0:6]]).all()
    assert (all[batches[1]] == indices[order[6:10]]).all()


def test_SubsetSampler_repeated():
    from batchup import sampling

    indices = np.random.RandomState(12345).permutation(20)[:10]
    spl = sampling.SubsetSampler(indices=indices, repeats=3)

    # Test index generation

    # In order
    ndx_iter = spl.in_order_indices_batch_iterator(batch_size=5)
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
    ndx_iter = spl.in_order_indices_batch_iterator(batch_size=7)
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
    shuffled_ndx_iter = spl.shuffled_indices_batch_iterator(
        batch_size=5, shuffle_rng=np.random.RandomState(12345))
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
    shuffled_ndx_iter = spl.shuffled_indices_batch_iterator(
        batch_size=7, shuffle_rng=np.random.RandomState(12345))
    batches = list(shuffled_ndx_iter)
    order_rng = np.random.RandomState(12345)
    order = np.concatenate([order_rng.permutation(10) for _ in range(3)])
    assert len(batches) == 5
    assert (all[batches[0]] == indices[order[0:7]]).all()
    assert (all[batches[1]] == indices[order[7:14]]).all()
    assert (all[batches[2]] == indices[order[14:21]]).all()
    assert (all[batches[3]] == indices[order[21:28]]).all()
    assert (all[batches[4]] == indices[order[28:30]]).all()

    # Check invalid values for repeats
    with pytest.raises(ValueError):
        _ = sampling.SubsetSampler(indices=indices, repeats=-5)


def test_SubsetSampler_repeated_small_dataset():
    from batchup import sampling

    indices = np.random.RandomState(12345).permutation(40)[:20]
    spl_inf = sampling.SubsetSampler(indices=indices, repeats=-1)
    assert spl_inf.num_indices_generated() == np.inf

    # Test index generation

    # In order
    ndx_iter = spl_inf.in_order_indices_batch_iterator(batch_size=64)
    all = np.arange(40)
    batches = [next(ndx_iter) for _ in range(10)]
    assert len(batches) == 10
    x = np.concatenate([all[b] for b in batches], axis=0)
    assert (x == np.tile(indices, [32])).all()

    # Shuffled
    shuffled_ndx_iter = spl_inf.shuffled_indices_batch_iterator(
        batch_size=64, shuffle_rng=np.random.RandomState(12345))
    batches = [next(shuffled_ndx_iter) for _ in range(10)]
    order_rng = np.random.RandomState(12345)
    order = np.concatenate([order_rng.permutation(20) for _ in range(32)])
    assert len(batches) == 10
    x = np.concatenate([all[b] for b in batches], axis=0)
    assert (x == indices[order]).all()

    # Test fixed number of repetitions
    spl_16 = sampling.SubsetSampler(indices=indices, repeats=16)
    assert spl_16.num_indices_generated() == 320

    # In order
    ndx_iter = spl_16.in_order_indices_batch_iterator(batch_size=64)
    all = np.arange(40)
    batches = list(ndx_iter)
    assert len(batches) == 5
    x = np.concatenate([all[b] for b in batches], axis=0)
    assert (x == np.tile(indices, [16])).all()

    # Shuffled
    shuffled_ndx_iter = spl_16.shuffled_indices_batch_iterator(
        batch_size=64, shuffle_rng=np.random.RandomState(12345))
    batches = list(shuffled_ndx_iter)
    order_rng = np.random.RandomState(12345)
    order = np.concatenate([order_rng.permutation(20) for _ in range(16)])
    assert len(batches) == 5
    x = np.concatenate([all[b] for b in batches], axis=0)
    assert (x == indices[order]).all()


def test_WeightedSampler():
    import itertools
    from batchup import sampling

    weights = np.array([0.1, 0.2, 0.3, 0.4])
    spl = sampling.WeightedSampler(weights=weights)

    # Infinite indices generated
    assert spl.num_indices_generated() == np.inf

    # Test index mapping
    indices_to_map = np.random.randint(0, 4, size=(20,))
    assert (spl.map_indices(indices_to_map) == indices_to_map).all()

    # In-order generation not supported
    with pytest.raises(NotImplementedError):
        _ = spl.in_order_indices_batch_iterator(batch_size=5)

    # Test index generation

    # Shuffled
    shuffled_ndx_iter = spl.shuffled_indices_batch_iterator(
        batch_size=5, shuffle_rng=np.random.RandomState(12345))
    batches = list(itertools.islice(shuffled_ndx_iter, 4))
    order = np.random.RandomState(12345).choice(4, size=(20,), p=weights)
    assert len(batches) == 4
    drawn = np.concatenate(batches, axis=0)
    assert (drawn == order).all()

    # Weights should not sum to 0
    with pytest.raises(ValueError):
        _ = sampling.WeightedSampler(weights=np.zeros((3,)))

    # Test class balancing helpers
    # Ground truths
    y = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
    weights = sampling.WeightedSampler.class_balancing_sample_weights(y)
    expected_weights = np.array([1.0/6.0]*2 + [1.0/9.0]*3 + [1.0/15.0]*5)
    assert (weights == expected_weights).all()

    # Test class balancing constructor
    cls_bal_spl = sampling.WeightedSampler.class_balancing_sampler(y)

    shuffled_ndx_iter = cls_bal_spl.shuffled_indices_batch_iterator(
        batch_size=10, shuffle_rng=np.random.RandomState(12345))
    batches = list(itertools.islice(shuffled_ndx_iter, 10))
    order = np.random.RandomState(12345).choice(
        10, size=(100,), p=expected_weights)
    assert len(batches) == 10
    drawn = np.concatenate(batches, axis=0)
    assert (drawn == order).all()


def test_WeightedSubsetSampler():
    import itertools
    from batchup import sampling

    indices = np.array([0, 2, 4])
    weights = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    sub_weights = weights[indices]
    nrm_sub_weights = sub_weights / sub_weights.sum()
    spl = sampling.WeightedSubsetSampler(sub_weights=sub_weights,
                                         indices=indices)

    # Test index mapping
    indices_to_map = np.random.randint(0, 3, size=(20,))
    assert (spl.map_indices(indices_to_map) == indices[indices_to_map]).all()

    # Infinite indices generated
    assert spl.num_indices_generated() == np.inf

    # In-order generation not supported
    with pytest.raises(NotImplementedError):
        _ = spl.in_order_indices_batch_iterator(batch_size=5)

    # Test index generation

    # Shuffled
    shuffled_ndx_iter = spl.shuffled_indices_batch_iterator(
        batch_size=5, shuffle_rng=np.random.RandomState(12345))
    batches = list(itertools.islice(shuffled_ndx_iter, 4))
    order = np.random.RandomState(12345).choice(3, size=(20,),
                                                p=nrm_sub_weights)
    assert len(batches) == 4
    drawn = np.concatenate(batches, axis=0)
    assert (drawn == indices[order]).all()

    # Should be same number of weights and samples
    with pytest.raises(ValueError):
        _ = sampling.WeightedSubsetSampler(sub_weights=np.ones((5,)),
                                           indices=np.arange(3))

    # Weights should not sum to 0
    with pytest.raises(ValueError):
        _ = sampling.WeightedSubsetSampler(sub_weights=np.zeros((3,)),
                                           indices=np.arange(3))

    # Test class balancing constructor
    # Ground truths
    y = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
    indices = np.array([0, 2, 3, 5, 6, 7])
    weights = sampling.WeightedSampler.class_balancing_sample_weights(y)
    sub_weights = weights[indices]
    sub_weights = sub_weights / sub_weights.sum()
    print(sub_weights)
    expected_weights = np.array([1.0/3.0]*1 + [1.0/6.0]*2 + [1.0/9.0]*3)
    print(sub_weights.sum(), expected_weights.sum())

    cls_bal_spl = sampling.WeightedSubsetSampler.class_balancing_sampler(
        y, indices)

    assert (expected_weights == cls_bal_spl.sub_weights).all()

    shuffled_ndx_iter = cls_bal_spl.shuffled_indices_batch_iterator(
        batch_size=10, shuffle_rng=np.random.RandomState(12345))
    batches = list(itertools.islice(shuffled_ndx_iter, 10))
    order = np.random.RandomState(12345).choice(
        indices, size=(100,), p=expected_weights)
    assert len(batches) == 10
    drawn = np.concatenate(batches, axis=0)
    assert (drawn == order).all()
