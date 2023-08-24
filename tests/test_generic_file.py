import inspect
from pathlib import Path

from magnetopy import GenericFile

TESTS_PATH = Path(inspect.getfile(inspect.currentframe())).parent
DATA_PATH = TESTS_PATH / "data"
MVSH1_PATH = DATA_PATH / "mvsh1.dat"

GENERIC_FILE = GenericFile(MVSH1_PATH)


class TestGenericFile:
    def test_generic_file_repr(self):
        assert repr(GENERIC_FILE) == "GenericFile(mvsh1.dat)"

    def test_generic_file_str(self):
        assert str(GENERIC_FILE) == "GenericFile(mvsh1.dat)"

    def test_generic_file_exp_type(self):
        assert GENERIC_FILE.experiment_type == ""

    def test_generic_file_local_path(self):
        assert GENERIC_FILE.local_path == MVSH1_PATH

    def test_generic_file_as_dict(self):
        serialized = GENERIC_FILE.as_dict()
        assert serialized["local_path"] == str(MVSH1_PATH)
