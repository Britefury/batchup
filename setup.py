import os
from setuptools import find_packages
from setuptools import setup

version = '0.2.1'

here = os.path.abspath(os.path.dirname(__file__))
try:
    README = open(os.path.join(here, 'README.md')).read()
except IOError:
    README = ''

install_requires = [
    'numpy',
    'scipy',
    'six',
    'tables',
    'joblib',
]

tests_require = [
    'mock',
    'pytest',
    'pytest-cov',
    'pytest-pep8',
]

setup(
    name="batchup",
    version=version,
    description="Python library for extracting mini-batches of data from a data source for the purpose of training neural networks",
    long_description="\n\n".join([README]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    keywords="",
    author="Geoffrey French",
    # author_email="brittix1023 at gmail dot com",
    url="https://github.com/Britefury/batchup",
    license="MIT",
    packages=find_packages(),
    include_package_data=False,
    zip_safe=False,
    install_requires=install_requires,
    extras_require={
        'testing': tests_require,
        },
    )
