import pytest


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", help="run slow tests")
    parser.addoption("--runbig", action="store_true",
                     help="run tests that require large datasets")


def pytest_runtest_setup(item):
    if 'slow' in item.keywords and not item.config.getoption("--runslow"):
        pytest.skip("need --runslow option to run")
    if 'bigdataset' in item.keywords and not item.config.getoption("--runbig"):
        pytest.skip("need --runbig option to run")
