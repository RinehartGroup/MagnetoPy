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

    def test_generic_file_length(self):
        assert GENERIC_FILE.length == 2612152

    def test_generic_file_sha512(self):
        assert GENERIC_FILE.sha512 == (
            "6fc436762a00b890eb3649eb50a885ced587781bf3b9738f04a49e768ad471f167111"
            "0f282e7be2ac2ed623a006abcc2da3914e09c165276b4bd63e06760b28f"
        )

    def test_generic_file_as_dict(self):
        serialized = GENERIC_FILE.as_dict()
        assert serialized["local_path"] == str(MVSH1_PATH)
        assert serialized["length"] == 2612152
        assert serialized["sha512"] == (
            "6fc436762a00b890eb3649eb50a885ced587781bf3b9738f04a49e768ad471f167111"
            "0f282e7be2ac2ed623a006abcc2da3914e09c165276b4bd63e06760b28f"
        )
