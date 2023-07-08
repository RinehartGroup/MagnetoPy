import inspect
from pathlib import Path

import pytest

import magnetopy as mp

tests_path = Path(inspect.getfile(inspect.currentframe())).parent
data_path = tests_path / "data"

MVSH5_PATH = data_path / "mvsh5.dat"


@pytest.fixture
def mvsh5() -> mp.QDFile:
    return mp.QDFile(MVSH5_PATH)


@pytest.fixture
def mvsh5_with_raw() -> mp.QDFile:
    return mp.QDFile(MVSH5_PATH, process_raw=True)
