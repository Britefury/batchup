import os, sys
if sys.version_info[0] == 2:
    from urllib import urlretrieve
    from ConfigParser import RawConfigParser
else:
    from urllib.request import urlretrieve
    from configparser import RawConfigParser

_CONFIG_PATH = os.path.expanduser(os.path.join('~', '.batchup.cfg'))
_DEFAULT_BATCHUP_PATH = os.path.expanduser(os.path.join('~', '.batchup'))
_DATA_DIR_NAME = 'datasets'

_config__ = None
def get_config():
    global _config__
    if _config__ is None:
        if os.path.exists(_CONFIG_PATH):
            try:
                _config__ = RawConfigParser.read(_CONFIG_PATH)
            except:
                print('batchup: WARNING: error trying to open config file from {}'.format(
                    _CONFIG_PATH))
                _config__ = RawConfigParser()
        else:
            _config__ = RawConfigParser()
    return _config__

_data_dir_path__ = None
def get_batchup_path():
    global _data_dir_path__
    if _data_dir_path__ is None:
        try:
            _data_dir_path__ = get_config().get('paths', 'data_dir')
        except:
            _data_dir_path__ = _DEFAULT_BATCHUP_PATH
        if os.path.exists(_data_dir_path__):
            if not os.path.isdir(_data_dir_path__):
                raise RuntimeError('batchup: the DATA directory path ({}) is not a '
                                   'directory'.format(_data_dir_path__))
        else:
            os.makedirs(_data_dir_path__)
    return _data_dir_path__


def get_dataset_dir():
    return os.path.join(get_batchup_path(), _DATA_DIR_NAME)


def download(path, source_url):
    dir_path = os.path.split(path)[0]
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if not os.path.exists(path):
        filename = source_url.split('/')[-1]
        def _progress(count, block_size, total_size):
            sys.stdout.write('\rDownloading {} {:.2%}'.format(filename, float(count * block_size) / float(total_size)))
            sys.stdout.flush()
        urlretrieve(source_url, path, reporthook=_progress)
    return path


def download_data(filename, source_url):
    data_dir = get_dataset_dir()
    return download(os.path.join(data_dir, filename), source_url)
