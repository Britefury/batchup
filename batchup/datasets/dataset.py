import os
from .. import config


class AbstractSourceFile (object):
    """
    Abstract source file
    """
    def __init__(self, filename, sha256=None):
        self.filename = filename
        self.temp_filename = os.path.join('temp', filename)
        self.path = config.get_data_path(self.temp_filename)
        self.sha256 = sha256

    def acquire(self, **kwargs):
        raise NotImplementedError('Not implemented for {}'.format(
            type(self)))


class DownloadSourceFile (AbstractSourceFile):
    """
    A downloadable source file for a dataset.

    Invoke the `acquire` method to retrieve the file to BatchUp's temporary
    directory. Will not download the file if it is already present.
    """
    def __init__(self, filename, url=None, base_url=None, sha256=None):
        """
        Constructor

        Parameters
        ----------
        filename: str
            The name of the source file
        url: str or None
            The URL from which to download the file. If `url` is given the
            value `None`, `base_url` must be provided.
        base_url: str or None
            Give `base_url` a value if `url` is None, in which case the URL
            will be the result of concatenating `base_url` and `filename`
        sha256: str or None
            The expected SHA-256 hex digest of the file. The digest of the
            file will be checked against this. If there is a mis-match, the
            downloaded file will be deleted and the download will be
            re-attempted.
        """
        super(DownloadSourceFile, self).__init__(filename, sha256=sha256)
        if url is None:
            if base_url is None:
                raise ValueError('Must provide either url or base_url')
            if base_url.endswith('/'):
                url = base_url + filename
            else:
                url = base_url + '/' + filename
        self.url = url

    def __str__(self):
        return 'downloadable file {} from {}'.format(self.filename, self.url)

    def acquire(self, **kwargs):
        """
        Download the file and return its path

        Returns
        -------
        str
            The path of the file in BatchUp's temporary directory
        """
        return config.download_data(self.temp_filename, self.url,
                                    self.sha256)


class CopySourceFile (AbstractSourceFile):
    """
    A source file on the file system for a dataset.

    Invoke the `acquire` method to copy the file to BatchUp's temporary
    directory. Will not copy the file if it is already present.
    """

    def __init__(self, filename, source_path=None, arg_name=None,
                 sha256=None):
        """
        Constructor

        Parameters
        ----------
        filename: str
            The name of the source file
        source_path: str or None
            The path from which to copy the file on the filesystem, or None
            to acquire it from arguments, in which case `arg_name` must
            be provided.
        arg_name: str or None
            The name of the argument supplied to the fetching function
            (see `fetch_and_convert_dataset` decorator) that will provide the
            path; used instead of source_path.
        sha256: str or None
            The expected SHA-256 hex digest of the file. The digest of the
            file will be checked against this. If there is a mis-match, the
            downloaded file will be deleted and the download will be
            re-attempted.
        """
        super(CopySourceFile, self).__init__(filename, sha256=sha256)
        if source_path is None and arg_name is None:
            raise ValueError('Either source_path or arg_name must be provided')
        self.source_path = source_path
        self.arg_name = arg_name

    def __str__(self):
        return 'file {} on filesystem at {}'.format(
            self.filename, self.source_path)

    def acquire(self, **kwargs):
        """
        Copy the file and return its path

        Returns
        -------
        str
            The path of the file in BatchUp's temporary directory
        """
        if self.source_path is None:
            source_path = kwargs[self.arg_name]
        else:
            source_path = self.source_path
        return config.copy_data(self.temp_filename, source_path, self.sha256)


def fetch_and_convert_dataset(source_files, target_filename):
    """
    Decorator applied to a dataset conversion function that converts acquired
    source files into a dataset file that BatchUp can use.

    Parameters
    ----------
    source_file: list of `AbstractSourceFile` instances
        A list of files to be acquired
    target_filename: str
        The name of the target file in which to store the converted data.

    The conversion function is of the form `fn(source_paths, target_path)`.
    It should return `target_path` if successful, `None` otherwise.
    After the conversion function is successfully applied, the temporary
    source files that were downloaded or copied into BatchUp's temporary
    directory are deleted, unless the conversion function moved or deleted
    them in which case no action is taken.

    Example
    -------
    In this example, we will show how to acquire the USPS dataset from an
    online source. USPS is provided as an HDF5 file anyway, so the
    conversion function simply moves it to the target path:

    >>> import shutil
    >>>
    >>> _USPS_SRC_ONLINE = DownloadSourceFile(
    ...    filename='usps.h5',
    ...    url='https://github.com/Britefury/usps_dataset/raw/master/'
    ...        'usps.h5',
    ...    sha256='ba768d9a9b11e79b31c1e40130647c4fc04e6afc1fb41a0d4b9f11'
    ...           '76065482b4'
    ... )
    >>>
    >>> @fetch_and_convert_dataset([_USPS_SRC_ONLINE], 'usps.h5')
    ... def usps_data_online(source_paths, target_path):
    ...    usps_path = source_paths[0]
    ...    # For other datasets, you would convert the data here
    ...    # In this case, we move the file
    ...    shutil.move(usps_path, target_path)
    ...    # Return the target path indicating success
    ...    return target_path
    >>>
    >>> # Now use it:
    >>> usps_path = usps_data_online() # doctest:+ELLIPSIS
    ...

    In this example, the USPS dataset will be acquired from a file on the
    filesystem. Note that the source path is fixed; the next example
    shows how we can determine the source path dynamically:

    >>> _USPS_SRC_OFFLINE_FIXED = CopySourceFile(
    ...    filename='usps.h5',
    ...    source_path='some/path/to/usps.h5',
    ...    sha256='ba768d9a9b11e79b31c1e40130647c4fc04e6afc1fb41a0d4b9f11'
    ...           '76065482b4'
    ... )
    >>>
    >>> @fetch_and_convert_dataset([_USPS_SRC_OFFLINE_FIXED], 'usps.h5')
    ... def usps_data_offline_fixed(source_paths, target_path):
    ...    usps_path = source_paths[0]
    ...    # For other datasets, you would convert the data here
    ...    # In this case, we move the file
    ...    shutil.move(usps_path, target_path)
    ...    # Return the target path indicating success
    ...    return target_path
    >>>
    >>> # Now use it:
    >>> usps_path = usps_data_offline_fixed() # doctest:+ELLIPSIS
    ...

    The source path is provided as an argument to the decorated fetch
    function:

    >>> _USPS_SRC_OFFLINE_DYNAMIC = CopySourceFile(
    ...    filename='usps.h5',
    ...    arg_name='usps_path',
    ...    sha256='ba768d9a9b11e79b31c1e40130647c4fc04e6afc1fb41a0d4b9f11'
    ...           '76065482b4'
    ... )
    >>>
    >>> @fetch_and_convert_dataset([_USPS_SRC_OFFLINE_DYNAMIC], 'usps.h5')
    ... def usps_data_offline_dynamic(source_paths, target_path):
    ...    usps_path = source_paths[0]
    ...    # For other datasets, you would convert the data here
    ...    # In this case, we move the file
    ...    shutil.move(usps_path, target_path)
    ...    # Return the target path indicating success
    ...    return target_path
    >>>
    >>> # Now use it (note that the KW-arg `usps_path` is the same
    >>> # as the `arg_name` parameter given to `CopySourceFile` above:
    >>> usps_path = usps_data_offline_dynamic(
    ...    usps_path='look/here.h5') # doctest:+ELLIPSIS
    ...
    """
    def decorate_fetcher(convert_function):
        def fetch(**kwargs):
            target_path = config.get_data_path(target_filename)

            # If the target file does not exist, we need to acquire the
            # source files and convert them
            if not os.path.exists(target_path):
                # Acquire the source files
                source_paths = []
                for src in source_files:
                    if not isinstance(src, AbstractSourceFile):
                        raise TypeError('source_files should contain'
                                        '`SourceFile` instances, '
                                        'not {}'.format(type(src)))
                    p = src.acquire(**kwargs)
                    if p is not None:
                        source_paths.append(p)
                    else:
                        print('Failed to acquire {}'.format(src))
                        return None

                # Got the source files
                # Convert
                converted_path = convert_function(source_paths, target_path)

                # If successful, delete the source files
                if converted_path is not None:
                    for p in source_paths:
                        if os.path.exists(p):
                            os.remove(p)

                return converted_path
            else:
                # Target file already exists
                return target_path

        fetch.__name__ = convert_function.__name__

        return fetch

    return decorate_fetcher


def delete_dataset_cache(*filenames):
    """
    Delete the cache (converted files) for a dataset.

    Parameters
    ----------
    filenames: str
        Filenames of files to delete
    """
    for filename in filenames:
        path = config.get_data_path(filename)
        if os.path.exists(path):
            os.remove(path)
