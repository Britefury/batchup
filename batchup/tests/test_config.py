import pytest


def _setup_batchup_path(monkeypatch, path):
    from batchup import config

    def get_batchup_path_patch():
        return path

    monkeypatch.setattr(config, 'get_batchup_path', get_batchup_path_patch)


def _setup_batchup_temp(monkeypatch):
    import tempfile

    tdir = tempfile.mkdtemp()

    _setup_batchup_path(monkeypatch, tdir)

    return tdir


def _teardown_batchup_temp(tdir):
    import shutil

    shutil.rmtree(tdir)


def _setup_batchup_temp_and_urlretrieve(monkeypatch, downloads=None):
    import tempfile
    from batchup import config

    tdir = _setup_batchup_temp(monkeypatch)

    def urlretrieve_patch(url, path, reporthook):
        with open(path, 'w') as f:
            f.write(url)
        if downloads is not None:
            downloads.append((url, path))
        reporthook(1, len(url), len(url))

    monkeypatch.setattr(config, 'urlretrieve', urlretrieve_patch)

    return tdir


def test_get_data_dir(monkeypatch):
    import os
    from batchup import config

    _setup_batchup_path(monkeypatch, 'test')

    assert config.get_data_dir() == os.path.join('test', 'data')


def test_get_data_path(monkeypatch):
    import os
    from batchup import config

    _setup_batchup_path(monkeypatch, 'test')

    assert config.get_data_path('foo.xyz') == os.path.join(
        'test', 'data', 'foo.xyz')
    assert config.get_data_path(os.path.abspath('foo.xyz')) == \
        os.path.abspath('foo.xyz')


def test_download(monkeypatch, capsys):
    import os
    from batchup import config

    _downloads = []

    tdir = _setup_batchup_temp_and_urlretrieve(monkeypatch, _downloads)

    download_path = os.path.join(tdir, 'a', 'b', 'c', 'd.h5')

    res = config.download(download_path, 'myurl/d.h5')
    out, err = capsys.readouterr()
    line1 = 'Downloading {} to {}'.format('myurl/d.h5', download_path)
    line2 = '\rDownloading {} {:.2%}'.format('d.h5', 1.0)
    assert out == '{}\n{}\r'.format(line1, line2)

    assert res == download_path
    assert len(_downloads) == 1
    assert _downloads[0][0] == 'myurl/d.h5'
    assert _downloads[0][1] == download_path

    assert os.path.exists(os.path.join(tdir, 'a', 'b', 'c'))
    assert os.path.exists(download_path)
    assert open(download_path, 'r').read() == 'myurl/d.h5'

    # Attempting to download a second time should exit
    res = config.download(download_path, 'myurl/d.h5')
    assert res == download_path
    assert len(_downloads) == 1
    out, err = capsys.readouterr()
    assert out == ''

    _teardown_batchup_temp(tdir)


def test_download_err(monkeypatch):
    import os
    import tempfile
    import shutil
    from batchup import config

    _downloads = []

    def urlretrieve_patch(url, path, reporthook):
        _downloads.append((url, path))
        with open(path, 'w') as f:
            f.write('hello world')
        raise ValueError

    tdir = tempfile.mkdtemp()

    _setup_batchup_path(monkeypatch, tdir)
    monkeypatch.setattr(config, 'urlretrieve', urlretrieve_patch)

    download_path = os.path.join(tdir, 'a', 'b', 'c', 'd.h5')

    with pytest.raises(ValueError):
        config.download(download_path, 'myurl/d.h5')

    assert _downloads[0][0] == 'myurl/d.h5'
    assert _downloads[0][1] == download_path

    assert os.path.exists(os.path.join(tdir, 'a', 'b', 'c'))
    # `download` should have removed the file
    assert not os.path.exists(download_path)

    shutil.rmtree(tdir)


def test_compute_sha256():
    import os
    import tempfile
    import shutil
    import hashlib
    from batchup import config

    tdir = tempfile.mkdtemp()
    fpath = os.path.join(tdir, 'hi.txt')

    with open(fpath, 'w') as f:
        f.write('hello world')

    hasher = hashlib.sha256()
    hasher.update(b'hello world')
    expected = hasher.hexdigest()

    assert config.compute_sha256(fpath) == expected

    shutil.rmtree(tdir)


def test_verify_file(capsys):
    import os
    import tempfile
    import shutil
    import hashlib
    from batchup import config

    tdir = tempfile.mkdtemp()
    fpath = os.path.join(tdir, 'hi.txt')
    invalid_path = os.path.join(tdir, 'bye.txt')

    with open(fpath, 'w') as f:
        f.write('hello world')

    hasher = hashlib.sha256()
    hasher.update(b'hello world')
    expected = hasher.hexdigest()

    bad_hasher = hashlib.sha256()
    bad_hasher.update(b'goodbye world')
    bad_expected = bad_hasher.hexdigest()

    # File should pass
    assert config.verify_file(fpath, expected)
    # Should not pass non-existent file
    assert not config.verify_file(invalid_path, expected)
    # Should not pass invalid hash
    assert not config.verify_file(fpath, bad_expected)

    # File should pass with *no* specified hash, but should generate output
    assert config.verify_file(fpath, None)
    out, err = capsys.readouterr()
    line1 = 'SHA-256 of {}:'.format(fpath)
    line2 = '  "{}"'.format(expected)
    assert out == '{}\n{}\n'.format(line1, line2)

    shutil.rmtree(tdir)


def test_download_data(monkeypatch, capsys):
    import os
    import tempfile
    import shutil
    import hashlib
    from batchup import config

    # Good and bad expected hashes
    hasher = hashlib.sha256()
    hasher.update(b'hello world')
    expected = hasher.hexdigest()

    _downloads = []
    _download_data = []

    def download_patch(path, url):
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        _downloads.append((url, path))
        data, err = _download_data[0]
        del _download_data[0]
        with open(path, 'w') as f:
            f.write(data)
        if err is not None:
            raise err()
        return path

    tdir = tempfile.mkdtemp()

    _setup_batchup_path(monkeypatch, tdir)
    monkeypatch.setattr(config, 'download', download_patch)

    # Successful download on first time
    _download_data[:] = [('hello world', None)]
    download_name = 'd.h5'
    download_path = config.get_data_path(download_name)

    res = config.download_data(download_name, 'myurl/d.h5', expected)
    out, err = capsys.readouterr()
    assert out == ''

    assert res == download_path
    assert len(_downloads) == 1
    assert _downloads[0] == ('myurl/d.h5', download_path + '.unverified')
    assert os.path.exists(download_path)
    assert open(download_path, 'r').read() == 'hello world'
    del _downloads[:]

    # Attempting to download a second time should exit
    _download_data[:] = [('hello world', None)]
    res = config.download_data(download_name, 'myurl/d.h5', expected)
    assert res == download_path
    assert len(_downloads) == 0
    out, err = capsys.readouterr()
    assert out == ''

    # Attempting to download a second time with no hash should print the hash
    _download_data[:] = [('hello world', None)]
    res = config.download_data(download_name, 'myurl/d.h5', None)
    assert res == download_path
    assert len(_downloads) == 0
    out, err = capsys.readouterr()
    assert out == 'The SHA-256 of {} is "{}"\n'.format(download_path, expected)

    os.remove(download_path)

    # Now test for download errors
    _download_data[:] = [
        ('', RuntimeError),     # Raise an exception on the 1st try
        ('goodbye world', None),    # Wrong data (fail verification) on 2nd
        ('hello world', None),
    ]
    res = config.download_data(download_name, 'myurl/d.h5', expected)
    out, err = capsys.readouterr()
    line1 = ('Download of {} unsuccessful; error {}; '
             'deleting and re-trying...'.format('myurl/d.h5', RuntimeError()))
    line2 = ('Download of {} unsuccessful; verification failed; '
             'deleting and re-trying...'.format('myurl/d.h5'))
    assert res == download_path
    assert len(_downloads) == 3
    assert out == '{}\n{}\n'.format(line1, line2)

    del _downloads[:]
    os.remove(download_path)

    # Now test for complete failure
    _download_data[:] = [
        ('', RuntimeError),     # Raise an exception on the 1st try
        ('', RuntimeError),     # Raise an exception on the 2nd try
        ('goodbye world', None),    # Wrong data (fail verification) on 2nd
    ]
    res = config.download_data(download_name, 'myurl/d.h5', expected)
    out, err = capsys.readouterr()
    line1 = ('Download of {} unsuccessful; error {}; '
             'deleting and re-trying...'.format('myurl/d.h5', RuntimeError()))
    line2 = ('Download of {} unsuccessful; error {}; '
             'deleting and re-trying...'.format('myurl/d.h5', RuntimeError()))
    line3 = ('Download of {} unsuccessful; verification failed; '
             'deleting and re-trying...'.format('myurl/d.h5'))
    line4 = ('Did not succeed in downloading {} (tried {} times)'.format(
             'myurl/d.h5', config._MAX_DOWNLOAD_TRIES))
    assert res is None
    assert len(_downloads) == 3
    assert out == '{}\n{}\n{}\n{}\n'.format(line1, line2, line3, line4)

    shutil.rmtree(tdir)


def test_copy_data(monkeypatch, capsys):
    import os
    import tempfile
    import shutil
    import hashlib
    from batchup import config

    _copy = shutil.copy
    _copies = []

    def copy_patch(source, dest):
        _copy(source, dest)
        _copies.append((source, dest))

    tdir = tempfile.mkdtemp()

    _setup_batchup_path(monkeypatch, tdir)
    monkeypatch.setattr(shutil, 'copy', copy_patch)

    # Good and bad expected hashes
    hasher = hashlib.sha256()
    hasher.update(b'hello world')
    expected = hasher.hexdigest()

    bad_hasher = hashlib.sha256()
    bad_hasher.update(b'goodbye world')
    bad_expected = bad_hasher.hexdigest()

    source_path = os.path.join(tdir, 'source.txt')
    dest_name = 'dest.txt'
    dest_path = config.get_data_path(dest_name)

    with open(source_path, 'w') as f:
        f.write('hello world')

    # Successful download on first time
    res = config.copy_data(dest_name, source_path, expected)
    out, err = capsys.readouterr()
    assert out == ''

    assert res == dest_path
    assert len(_copies) == 1
    assert _copies[0] == (source_path, dest_path + '.unverified')
    assert os.path.exists(dest_path)
    assert open(dest_path, 'r').read() == 'hello world'
    out, err = capsys.readouterr()
    assert out == ''
    del _copies[:]

    # Attempting to download a second time should exit
    res = config.copy_and_verify(dest_path, source_path, expected)
    assert res == dest_path
    assert len(_copies) == 0
    out, err = capsys.readouterr()
    assert out == ''

    # Attempting to download a second time with no hash should print the hash
    res = config.copy_and_verify(dest_path, source_path, None)
    assert res == dest_path
    assert len(_copies) == 0
    out, err = capsys.readouterr()
    assert out == 'The SHA-256 of {} is "{}"\n'.format(dest_path, expected)

    os.remove(dest_path)
    del _copies[:]

    # Try failing verification
    res = config.copy_and_verify(dest_path, source_path, bad_expected)
    out, err = capsys.readouterr()
    assert out == 'SHA verification of file {} failed\n'.format(source_path)

    assert res is None
    assert len(_copies) == 1
    assert _copies[0] == (source_path, dest_path + '.unverified')
    assert not os.path.exists(dest_path)
    del _copies[:]

    shutil.rmtree(tdir)
