import os
import sys
import hashlib
import shutil
if sys.version_info[0] == 2:  # pragma: no cover
    from urllib import urlretrieve
    from ConfigParser import RawConfigParser
else:
    from urllib.request import urlretrieve
    from configparser import RawConfigParser

_CONFIG_PATH = os.path.expanduser(os.path.join('~', '.batchup.cfg'))
_DEFAULT_BATCHUP_PATH = os.path.expanduser(os.path.join('~', '.batchup'))
_BATCHUP_ENV_NAME = 'BATCHUP_HOME'
_DATA_DIR_NAME = 'data'
_MAX_DOWNLOAD_TRIES = 3

_config__ = None
_data_dir_path__ = None


def get_config():  # pragma: no cover
    global _config__
    if _config__ is None:
        if os.path.exists(_CONFIG_PATH):
            try:
                _config__ = RawConfigParser()
                _config__.read(_CONFIG_PATH)
            except Exception as e:
                print('batchup: WARNING: error {} trying to open config '
                      'file from {}'.format(e, _CONFIG_PATH))
                _config__ = RawConfigParser()
        else:
            _config__ = RawConfigParser()
    return _config__


def get_batchup_path():  # pragma: no cover
    global _data_dir_path__
    if _data_dir_path__ is None:
        try:
            _data_dir_path__ = get_config().get('paths', 'data_dir')
        except:
            _data_dir_path__ = os.environ.get(_BATCHUP_ENV_NAME,
                                              _DEFAULT_BATCHUP_PATH)
        if os.path.exists(_data_dir_path__):
            if not os.path.isdir(_data_dir_path__):
                raise RuntimeError(
                    'batchup: the DATA directory path ({}) is not a '
                    'directory'.format(_data_dir_path__))
        else:
            os.makedirs(_data_dir_path__)
    return _data_dir_path__


def get_data_dir():
    """
    Get the batchup data directory path

    Returns
    -------
    str
        The path of the batchup data directory
    """
    return os.path.join(get_batchup_path(), _DATA_DIR_NAME)


def get_data_path(filename):
    """
    Get the path of the given file within the batchup data directory

    Parameters
    ----------
    filename: str
        The filename to locate within the batchup data directory

    Returns
    -------
    str
        The full path of the file
    """
    if os.path.isabs(filename):
        return filename
    else:
        return os.path.join(get_data_dir(), filename)


def download(path, source_url):
    """
    Download a file to a given path from a given URL, if it does not exist.

    Parameters
    ----------
    path: str
        The (destination) path of the file on the local filesystem
    source_url: str
        The URL from which to download the file

    Returns
    -------
    str
        The path of the file
    """
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if not os.path.exists(path):
        print('Downloading {} to {}'.format(source_url, path))
        filename = source_url.split('/')[-1]

        def _progress(count, block_size, total_size):
            sys.stdout.write('\rDownloading {} {:.2%}'.format(
                filename, float(count * block_size) / float(total_size)))
            sys.stdout.flush()
        try:
            urlretrieve(source_url, path, reporthook=_progress)
        except:
            sys.stdout.write('\r')
            # Exception; remove any partially downloaded file and re-raise
            if os.path.exists(path):
                os.remove(path)
            raise
        sys.stdout.write('\r')
    return path


def compute_sha256(path):
    """
    Compute the SHA-256 hash of the file at the given path

    Parameters
    ----------
    path: str
        The path of the file

    Returns
    -------
    str
        The SHA-256 HEX digest
    """
    hasher = hashlib.sha256()
    with open(path, 'rb') as f:
        # 10MB chunks
        for chunk in iter(lambda: f.read(10 * 1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_file(path, sha256):
    """
    Verify the integrity of a file by checking its SHA-256 hash.
    If no digest is supplied, the digest is printed to the console.

    Closely follows the code in `torchvision.datasets.utils.check_integrity`

    Parameters
    ----------
    path: str
        The path of the file to check
    sha256: str
        The expected SHA-256 hex digest of the file, or `None` to print the
        digest of the file to the console

    Returns
    -------
    bool
        Indicates if the file passes the integrity check or not
    """
    if not os.path.isfile(path):
        return False
    digest = compute_sha256(path)
    if sha256 is None:
        # No digest supplied; report it to the console so a develop can fill
        # it in
        print('SHA-256 of {}:'.format(path))
        print('  "{}"'.format(digest))
    else:
        if digest != sha256:
            return False
    return True


def download_and_verify(path, source_url, sha256):
    """
    Download a file to a given path from a given URL, if it does not exist.
    After downloading it, verify it integrity by checking the SHA-256 hash.

    Parameters
    ----------
    path: str
        The (destination) path of the file on the local filesystem
    source_url: str
        The URL from which to download the file
    sha256: str
        The expected SHA-256 hex digest of the file, or `None` to print the
        digest of the file to the console

    Returns
    -------
    str or None
        The path of the file if successfully downloaded otherwise `None`
    """
    if os.path.exists(path):
        # Already exists?
        # Nothing to do, except print the SHA-256 if necessary
        if sha256 is None:
            print('The SHA-256 of {} is "{}"'.format(
                path, compute_sha256(path)))
        return path

    # Compute the path of the unverified file
    unverified_path = path + '.unverified'
    for i in range(_MAX_DOWNLOAD_TRIES):
        # Download it
        try:
            unverified_path = download(unverified_path, source_url)
        except Exception as e:
            # Report failure
            print(
                'Download of {} unsuccessful; error {}; '
                'deleting and re-trying...'.format(source_url, e))
            # Delete so that we can retry
            if os.path.exists(unverified_path):
                os.remove(unverified_path)
        else:
            if os.path.exists(unverified_path):
                # Got something...
                if verify_file(unverified_path, sha256):
                    # Success: rename the unverified file to the destination
                    # filename
                    os.rename(unverified_path, path)
                    return path
                else:
                    # Report failure
                    print(
                        'Download of {} unsuccessful; verification failed; '
                        'deleting and re-trying...'.format(source_url))
                    # Delete so that we can retry
                    os.remove(unverified_path)

    print('Did not succeed in downloading {} (tried {} times)'.format(
        source_url, _MAX_DOWNLOAD_TRIES
    ))
    return None


def copy_and_verify(path, source_path, sha256):
    """
    Copy a file to a given path from a given path, if it does not exist.
    After copying it, verify it integrity by checking the SHA-256 hash.

    Parameters
    ----------
    path: str
        The (destination) path of the file on the local filesystem
    source_path: str
        The path from which to copy the file
    sha256: str
        The expected SHA-256 hex digest of the file, or `None` to print the
        digest of the file to the console

    Returns
    -------
    str or None
        The path of the file if successfully downloaded otherwise `None`
    """
    if os.path.exists(path):
        # Already exists?
        # Nothing to do, except print the SHA-256 if necessary
        if sha256 is None:
            print('The SHA-256 of {} is "{}"'.format(
                path, compute_sha256(path)))
        return path

    if not os.path.exists(source_path):
        return None

    # Compute the path of the unverified file
    unverified_path = path + '.unverified'
    # Copy it
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    shutil.copy(source_path, unverified_path)

    if os.path.exists(unverified_path):
        # Got something...
        if verify_file(unverified_path, sha256):
            # Success: rename the unverified file to the destination
            # filename
            os.rename(unverified_path, path)
            return path
        else:
            # Report failure
            print('SHA verification of file {} failed'.format(source_path))
            # Delete
            os.remove(unverified_path)
    return None


def download_data(filename, source_url, sha256):
    """
    Download a file into the BatchUp data directory from a given URL,
    if it does not exist.
    After downloading it, verify it integrity by checking the SHA-256 hash.

    Parameters
    ----------
    path: str
        The (destination) path of the file on the local filesystem
    source_url: str
        The URL from which to download the file
    sha256: str
        The expected SHA-256 hex digest of the file, or `None` to print the
        digest of the file to the console

    Returns
    -------
    str or None
        The path of the file if successfully downloaded otherwise `None`
    """
    return download_and_verify(get_data_path(filename), source_url, sha256)


def copy_data(filename, source_path, sha256):
    """
    Copy a file into the BatchUp data directory from a given path, if it
    does not exist. After copying it, verify it integrity by checking the
    SHA-256 hash.

    Parameters
    ----------
    path: str
        The (destination) path of the file on the local filesystem
    source_path: str
        The path from which to copy the file
    sha256: str
        The expected SHA-256 hex digest of the file, or `None` to print the
        digest of the file to the console

    Returns
    -------
    str or None
        The path of the file if successfully downloaded otherwise `None`
    """
    return copy_and_verify(get_data_path(filename), source_path, sha256)
