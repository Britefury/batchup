import hashlib


def sample_hashes(x):
    """
    Helper function that creates MD5 hashes of samples from a NumPy array.

    Parameters
    ----------
    x: a NumPuy array (or compatible, e.g. PyTables array)
        The array to draw samples

    Returns
    -------
    list
        A list of MD5 hashes of the samples in the array
    """
    hashes = []
    for i in range(len(x)):
        b = x[i].tobytes()
        h = hashlib.md5()
        h.update(b)
        hashes.append(h.hexdigest())
    return hashes
