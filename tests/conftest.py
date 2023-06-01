import inspect
from pathlib import Path

import pytest

import magnetopy as mp

tests_path = Path(inspect.getfile(inspect.currentframe())).parent
data_path = tests_path / "data"

MVSH1_PATH = data_path / "mvsh1.dat"


@pytest.fixture
def mvsh1() -> mp.QDFile:
    return mp.QDFile(MVSH1_PATH)


@pytest.fixture
def mvsh1_with_raw() -> mp.QDFile:
    return mp.QDFile(MVSH1_PATH, process_raw=True)
