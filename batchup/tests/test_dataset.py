import pytest
import os
import numpy as np
from . import test_config


def _get_data_dir():
    return os.path.abspath('some_data_dir')


def _patch_config_datadir(monkeypatch):
    from batchup import config

    monkeypatch.setattr(config, 'get_data_dir', _get_data_dir)


def test_path_string():
    from batchup.datasets import dataset

    assert dataset.path_string('foo') == 'foo'
    assert dataset.path_string(lambda: 'bar') == 'bar'

    with pytest.raises(TypeError):
        dataset.path_string(1)


def test_DownloadSourceFile_constructor(monkeypatch):
    from batchup.datasets import dataset

    _patch_config_datadir(monkeypatch)

    f1 = dataset.DownloadSourceFile(
        'test.txt', url='http://someplace.com/other.txt')
    assert f1.filename == 'test.txt'
    assert f1.temp_filename == os.path.join('temp', 'test.txt')
    assert f1.path == os.path.join(_get_data_dir(), f1.temp_filename)
    assert f1.url == 'http://someplace.com/other.txt'
    assert str(f1) == \
        'downloadable file test.txt from http://someplace.com/other.txt'

    f2 = dataset.DownloadSourceFile(
        'test.txt', base_url='http://someplace.com')
    assert f2.filename == 'test.txt'
    assert f2.temp_filename == os.path.join('temp', 'test.txt')
    assert f2.path == os.path.join(_get_data_dir(), f1.temp_filename)
    assert f2.url == 'http://someplace.com/test.txt'

    with pytest.raises(TypeError):
        dataset.DownloadSourceFile('test.txt')


def test_DownloadSourceFile_acquire(monkeypatch):
    from batchup.datasets import dataset
    import hashlib

    tdir = test_config._setup_batchup_temp_and_urlretrieve(monkeypatch)

    hasher = hashlib.sha256()
    hasher.update(b'http://someplace.com/other.txt')
    expected_sha = hasher.hexdigest()

    f1 = dataset.DownloadSourceFile(
        'test.txt', url='http://someplace.com/other.txt',
        sha256=expected_sha)
    assert f1.filename == 'test.txt'
    assert f1.temp_filename == os.path.join('temp', 'test.txt')
    assert f1.path == os.path.join(tdir, 'data', 'temp', 'test.txt')
    assert f1.url == 'http://someplace.com/other.txt'
    assert str(f1) == \
        'downloadable file test.txt from http://someplace.com/other.txt'

    dest = f1.acquire()
    assert dest == os.path.join(tdir, 'data', 'temp', 'test.txt')

    assert os.path.exists(dest)

    # clean up
    f1.clean_up()
    assert not os.path.exists(dest)

    test_config._teardown_batchup_temp(tdir)


def test_CopySourceFile_constructor(monkeypatch):
    from batchup.datasets import dataset

    _patch_config_datadir(monkeypatch)

    f1 = dataset.CopySourceFile(
        'test.txt', source_path=os.path.join('some_place', 'other.txt'))
    assert f1.filename == 'test.txt'
    assert f1.temp_filename == os.path.join('temp', 'test.txt')
    assert f1.path == os.path.join(_get_data_dir(), f1.temp_filename)
    assert f1.source_path == os.path.join('some_place', 'other.txt')
    assert f1.arg_name is None
    assert str(f1) == 'file test.txt on filesystem at {}'.format(
        os.path.join('some_place', 'other.txt'))

    f2 = dataset.CopySourceFile(
        'test.txt', arg_name='x')
    assert f2.filename == 'test.txt'
    assert f2.temp_filename == os.path.join('temp', 'test.txt')
    assert f2.path == os.path.join(_get_data_dir(), f1.temp_filename)
    assert f2.source_path is None
    assert f2.arg_name == 'x'

    with pytest.raises(TypeError):
        dataset.CopySourceFile('test.txt')


def test_CopySourceFile_acquire(monkeypatch):
    from batchup.datasets import dataset
    import hashlib

    tdir = test_config._setup_batchup_temp(monkeypatch)

    source_path = os.path.join(tdir, 'source.txt')
    with open(source_path, 'w') as f:
        f.write('hello world')

    hasher = hashlib.sha256()
    hasher.update(b'hello world')
    expected_sha = hasher.hexdigest()

    f1 = dataset.CopySourceFile(
        'test.txt', source_path=source_path,
        sha256=expected_sha)
    assert f1.filename == 'test.txt'
    assert f1.temp_filename == os.path.join('temp', 'test.txt')
    assert f1.path == os.path.join(tdir, 'data', 'temp', 'test.txt')
    assert f1.source_path == source_path
    assert f1.arg_name is None

    dest = f1.acquire()
    assert dest == os.path.join(tdir, 'data', 'temp', 'test.txt')

    assert os.path.exists(dest)
    assert open(dest, 'r').read() == 'hello world'

    # clean up
    f1.clean_up()
    assert not os.path.exists(dest)

    test_config._teardown_batchup_temp(tdir)


def test_CopySourceFile_acquire_arg(monkeypatch):
    from batchup.datasets import dataset
    import hashlib

    tdir = test_config._setup_batchup_temp(monkeypatch)

    source_path = os.path.join(tdir, 'source.txt')
    with open(source_path, 'w') as f:
        f.write('hello world')

    hasher = hashlib.sha256()
    hasher.update(b'hello world')
    expected_sha = hasher.hexdigest()

    f1 = dataset.CopySourceFile(
        'test.txt', arg_name='x',
        sha256=expected_sha)
    assert f1.filename == 'test.txt'
    assert f1.temp_filename == os.path.join('temp', 'test.txt')
    assert f1.path == os.path.join(tdir, 'data', 'temp', 'test.txt')
    assert f1.source_path is None
    assert f1.arg_name == 'x'

    dest = f1.acquire(x=source_path)
    assert dest == os.path.join(tdir, 'data', 'temp', 'test.txt')

    assert os.path.exists(dest)
    assert open(dest, 'r').read() == 'hello world'

    # clean up
    f1.clean_up()
    assert not os.path.exists(dest)

    test_config._teardown_batchup_temp(tdir)


def test_ExistingSourceFile_constructor(monkeypatch):
    from batchup.datasets import dataset

    _patch_config_datadir(monkeypatch)

    f1 = dataset.ExistingSourceFile(
        path=os.path.join('some_place', 'other.txt'))
    assert f1.path == os.path.join('some_place', 'other.txt')
    assert str(f1) == 'file at {}'.format(
        os.path.join('some_place', 'other.txt'))

    f2 = dataset.ExistingSourceFile(
        path=lambda: os.path.join('some_place', 'other.txt'))
    assert f1.path == os.path.join('some_place', 'other.txt')
    assert str(f1) == 'file at {}'.format(
        os.path.join('some_place', 'other.txt'))

    with pytest.raises(TypeError):
        dataset.ExistingSourceFile(1)


def test_ExistingSourceFile_acquire(monkeypatch):
    from batchup.datasets import dataset
    import hashlib

    tdir = test_config._setup_batchup_temp(monkeypatch)

    source_path = os.path.join(tdir, 'source.txt')
    with open(source_path, 'w') as f:
        f.write('hello world')

    hasher = hashlib.sha256()
    hasher.update(b'hello world')
    expected_sha = hasher.hexdigest()

    f1 = dataset.ExistingSourceFile(source_path, sha256=expected_sha)
    assert f1.path == source_path

    dest = f1.acquire()
    assert dest == source_path

    assert os.path.exists(dest)
    assert open(dest, 'r').read() == 'hello world'

    # clean up - should NOT remove the file
    f1.clean_up()
    assert os.path.exists(dest)

    test_config._teardown_batchup_temp(tdir)


def test_ExistingSourceFile_acquire_nonexistant(monkeypatch):
    from batchup.datasets import dataset
    import hashlib

    tdir = test_config._setup_batchup_temp(monkeypatch)

    source_path = os.path.join(tdir, 'source.txt')

    hasher = hashlib.sha256()
    hasher.update(b'hello world')
    expected_sha = hasher.hexdigest()

    f1 = dataset.ExistingSourceFile(source_path, sha256=expected_sha)
    assert f1.path == source_path

    dest = f1.acquire()
    assert dest is None

    test_config._teardown_batchup_temp(tdir)


def test_downloaded_dataset(monkeypatch):
    from batchup.datasets import dataset
    import hashlib

    tdir = test_config._setup_batchup_temp_and_urlretrieve(monkeypatch)

    hasher = hashlib.sha256()
    hasher.update(b'http://someplace.com/other.txt')
    expected_sha_a = hasher.hexdigest()
    f1 = dataset.DownloadSourceFile(
        'test.txt', url='http://someplace.com/other.txt',
        sha256=expected_sha_a)

    hasher = hashlib.sha256()
    hasher.update(b'http://someplace.com/somethingelse.txt')
    expected_sha_b = hasher.hexdigest()
    f2 = dataset.DownloadSourceFile(
        'test2.txt', url='http://someplace.com/somethingelse.txt',
        sha256=expected_sha_b)

    # Target filename (last arg) must be a string or a callable
    with pytest.raises(TypeError):
        @dataset.fetch_and_convert_dataset([f1, f2], 2)
        def downloaded_dataset(source_paths, target_path):
            raise RuntimeError('Should not get here')

    # Source files must contain `AbstractSourceFile` instances
    with pytest.raises(TypeError):
        @dataset.fetch_and_convert_dataset([f1, 'test2.txt'], 'ds.txt')
        def downloaded_dataset(source_paths, target_path):
            raise RuntimeError('Should not get here')

    @dataset.fetch_and_convert_dataset([f1, f2], 'ds.txt')
    def downloaded_dataset(source_paths, target_path):
        p1, p2 = source_paths
        with open(target_path, 'w') as f_out:
            f_out.write(open(p1, 'r').read())
            f_out.write(open(p2, 'r').read())
        return target_path

    dest = downloaded_dataset()

    # Check the resulting file
    assert os.path.exists(dest)
    assert open(dest, 'r').read() == (f1.url + f2.url)

    # Ensure that the temporary 'downloaded' files have been cleaned up
    assert not os.path.exists(f1.path)
    assert not os.path.exists(f2.path)

    # Invoking a second time should re-use the existing file
    dest2 = downloaded_dataset()
    assert dest2 == dest

    test_config._teardown_batchup_temp(tdir)


def test_downloaded_dataset_duplicate_sources(monkeypatch):
    from batchup.datasets import dataset
    import hashlib

    tdir = test_config._setup_batchup_temp_and_urlretrieve(monkeypatch)

    hasher = hashlib.sha256()
    hasher.update(b'http://someplace.com/other.txt')
    expected_sha_a = hasher.hexdigest()
    f1 = dataset.DownloadSourceFile(
        'test.txt', url='http://someplace.com/other.txt',
        sha256=expected_sha_a)

    hasher = hashlib.sha256()
    hasher.update(b'http://someplace.com/somethingelse.txt')
    expected_sha_b = hasher.hexdigest()
    f2 = dataset.DownloadSourceFile(
        'test.txt', url='http://someplace.com/somethingelse.txt',
        sha256=expected_sha_b)

    @dataset.fetch_and_convert_dataset([f1, f2], 'ds.txt')
    def downloaded_dataset(source_paths, target_path):
        raise RuntimeError('Should not get here')

    with pytest.raises(ValueError):
        downloaded_dataset()

    test_config._teardown_batchup_temp(tdir)


def test_copied_dataset(monkeypatch):
    from batchup.datasets import dataset
    import hashlib

    tdir = test_config._setup_batchup_temp(monkeypatch)

    source_a_path = os.path.join(tdir, 'source_a.txt')
    with open(source_a_path, 'w') as f:
        f.write('hello world')

    source_b_path = os.path.join(tdir, 'source_b.txt')

    # First file; CopiedSourceFile as normal
    hasher = hashlib.sha256()
    hasher.update(b'hello world')
    expected_sha_a = hasher.hexdigest()
    f1 = dataset.CopySourceFile(
        'test.txt', source_path=source_a_path,
        sha256=expected_sha_a)

    # Second file; CopiedSourceFile that gets the source path from the
    # argument 'x'
    hasher = hashlib.sha256()
    hasher.update(b', goodbye world')
    expected_sha_b = hasher.hexdigest()
    f2 = dataset.CopySourceFile(
        'test2.txt', arg_name='x',
        sha256=expected_sha_b)

    @dataset.fetch_and_convert_dataset([f1, f2], 'ds.txt')
    def copied_dataset(source_paths, target_path):
        p1, p2 = source_paths
        with open(target_path, 'w') as f_out:
            f_out.write(open(p1, 'r').read())
            f_out.write(open(p2, 'r').read())
        return target_path

    # The second file doesn't exist yet; should fail and return None
    # Pass a value for the argument 'x' telling BatchUp where to get
    # the f2 file from
    dest = copied_dataset(x=source_b_path)

    assert dest is None

    # The temporary file for f1 should exist to allow future attempts to
    # avoid having to re-acquire it
    assert os.path.exists(f1.path)
    assert not os.path.exists(f2.path)

    # === Create the second file and try again ===
    with open(source_b_path, 'w') as f:
        f.write(', goodbye world')

    # Pass a value for the argument 'x' telling BatchUp where to get
    # the f2 file from
    dest = copied_dataset(x=source_b_path)

    # Check the resulting file
    assert os.path.exists(dest)
    assert open(dest, 'r').read() == 'hello world, goodbye world'

    # Ensure that the temporary copied files have been cleaned up
    assert not os.path.exists(f1.path)
    assert not os.path.exists(f2.path)

    test_config._teardown_batchup_temp(tdir)


def test_existing_dataset(monkeypatch):
    from batchup.datasets import dataset
    import hashlib

    tdir = test_config._setup_batchup_temp(monkeypatch)

    source_path = os.path.join(tdir, 'source.txt')
    with open(source_path, 'w') as f:
        f.write('hello world')

    hasher = hashlib.sha256()
    hasher.update(b'hello world')
    expected_sha = hasher.hexdigest()
    f1 = dataset.ExistingSourceFile(source_path, sha256=expected_sha)

    @dataset.fetch_and_convert_dataset([f1], 'ds.txt')
    def existing_dataset(source_paths, target_path):
        return source_paths[0]

    dest = existing_dataset()

    # Check the resulting file
    assert dest == source_path
    assert open(dest, 'r').read() == 'hello world'

    test_config._teardown_batchup_temp(tdir)


def test_delete_dataset_cache(monkeypatch):
    from batchup.datasets import dataset
    import hashlib

    tdir = test_config._setup_batchup_temp_and_urlretrieve(monkeypatch)

    hasher = hashlib.sha256()
    hasher.update(b'http://someplace.com/other.txt')
    expected_sha_a = hasher.hexdigest()
    f1 = dataset.DownloadSourceFile(
        'test.txt', url='http://someplace.com/other.txt',
        sha256=expected_sha_a)

    hasher = hashlib.sha256()
    hasher.update(b'http://someplace.com/somethingelse.txt')
    expected_sha_b = hasher.hexdigest()
    f2 = dataset.DownloadSourceFile(
        'test2.txt', url='http://someplace.com/somethingelse.txt',
        sha256=expected_sha_b)

    @dataset.fetch_and_convert_dataset([f1, f2], 'ds.txt')
    def downloaded_dataset(source_paths, target_path):
        p1, p2 = source_paths
        with open(target_path, 'w') as f_out:
            f_out.write(open(p1, 'r').read())
            f_out.write(open(p2, 'r').read())
        return target_path

    dest = downloaded_dataset()

    # Check the resulting file
    assert os.path.exists(dest)
    assert open(dest, 'r').read() == (f1.url + f2.url)

    # Ensure that the temporary 'downloaded' files have been cleaned up
    assert not os.path.exists(f1.path)
    assert not os.path.exists(f2.path)

    # Delete the dataset cache; provide the filename
    dataset.delete_dataset_cache('ds.txt')
    assert not os.path.exists(dest)

    test_config._teardown_batchup_temp(tdir)
