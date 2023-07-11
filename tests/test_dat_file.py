import inspect
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import pytest

from magnetopy import DatFile

TESTS_PATH = Path(inspect.getfile(inspect.currentframe())).parent
DATA_PATH = TESTS_PATH / "data"

mvsh1 = DatFile(DATA_PATH / "mvsh1.dat")
mvsh2 = DatFile(DATA_PATH / "mvsh2.dat")
mvsh2a = DatFile(DATA_PATH / "mvsh2a.dat")
mvsh2b = DatFile(DATA_PATH / "mvsh2b.dat")
mvsh3 = DatFile(DATA_PATH / "mvsh3.dat")
mvsh4 = DatFile(DATA_PATH / "mvsh4.dat")
mvsh5 = DatFile(DATA_PATH / "mvsh5.dat")
mvsh5rw = DatFile(DATA_PATH / "mvsh5.rw.dat")
mvsh6 = DatFile(DATA_PATH / "mvsh6.dat")
mvsh7 = DatFile(DATA_PATH / "mvsh7.dat")
zfcfc1 = DatFile(DATA_PATH / "zfcfc1.dat")
zfcfc2 = DatFile(DATA_PATH / "zfcfc2.dat")
zfcfc3 = DatFile(DATA_PATH / "zfcfc3.dat")
zfcfc4 = DatFile(DATA_PATH / "zfcfc4.dat")
fc4a = DatFile(DATA_PATH / "fc4a.dat")
fc4b = DatFile(DATA_PATH / "fc4b.dat")
zfc4a = DatFile(DATA_PATH / "zfc4a.dat")
zfc4b = DatFile(DATA_PATH / "zfc4b.dat")
dataset4 = DatFile(DATA_PATH / "dataset4.dat")
pd_std1 = DatFile(DATA_PATH / "Pd_std1.dat")


class TestDatFileBaseAttrs:
    """This is essentially a repeat of the GenericFile test and will only look at
    attributes created by the GenericFile base class. Note that since DatFile
    overwrites the date_created attribute, it won't be included in this test.
    """

    def test_generic_file_repr(self):
        assert repr(mvsh1) == "DatFile(mvsh1.dat)"

    def test_generic_file_str(self):
        assert str(mvsh1) == "DatFile(mvsh1.dat)"

    def test_generic_file_exp_type(self):
        assert mvsh1.experiment_type == "magnetometry"

    def test_generic_file_local_path(self):
        assert mvsh1.local_path == Path(DATA_PATH / "mvsh1.dat")

    def test_generic_file_length(self):
        assert mvsh1.length == 2612152

    def test_generic_file_sha512(self):
        assert mvsh1.sha512 == (
            "6fc436762a00b890eb3649eb50a885ced587781bf3b9738f04a49e768ad471f167111"
            "0f282e7be2ac2ed623a006abcc2da3914e09c165276b4bd63e06760b28f"
        )

    def test_generic_file_as_dict(self):
        serialized = mvsh1.as_dict()
        assert serialized["local_path"] == str(Path(DATA_PATH / "mvsh1.dat"))
        assert serialized["length"] == 2612152
        assert serialized["sha512"] == (
            "6fc436762a00b890eb3649eb50a885ced587781bf3b9738f04a49e768ad471f167111"
            "0f282e7be2ac2ed623a006abcc2da3914e09c165276b4bd63e06760b28f"
        )


@dataclass
class Expected:
    experiment = "magnetometry"
    num_comments: int
    shape: tuple[int, int]
    date: str
    exps: list[str]


parameterized = [
    (mvsh1, Expected(0, (7305, 89), "2020-07-11T11:07:00", ["mvsh"])),
    (mvsh2, Expected(0, (719, 89), "2022-03-25T14:11:00", ["mvsh"])),
    (mvsh2a, Expected(0, (274, 89), "2022-03-25T14:11:00", ["mvsh"])),
    (mvsh2b, Expected(0, (445, 89), "2022-03-25T14:11:00", ["mvsh"])),
    (mvsh3, Expected(0, (141, 89), "2019-09-21T01:57:00", ["mvsh"])),
    (mvsh4, Expected(1, (230, 89), "2022-05-03T22:44:00", ["mvsh"])),
    (mvsh5, Expected(1, (230, 89), "2022-10-10T23:44:00", ["mvsh"])),
    (mvsh5rw, Expected(458, (138316, 7), "2022-10-11T00:00:00", [])),
    (mvsh6, Expected(0, (445, 89), "2021-10-02T21:00:00", ["mvsh"])),
    (mvsh7, Expected(0, (445, 89), "2022-08-24T08:08:00", ["mvsh"])),
    (zfcfc1, Expected(0, (504, 89), "2021-09-18T19:46:00", ["zfcfc"])),
    (zfcfc2, Expected(0, (494, 89), "2021-09-06T23:20:00", ["zfcfc"])),
    (zfcfc3, Expected(0, (513, 89), "2022-11-10T17:19:00", ["zfcfc"])),
    (
        zfcfc4,
        Expected(4, (7531, 89), "2022-05-03T15:22:00", ["zfc", "fc", "zfc", "fc"]),
    ),
    (fc4a, Expected(1, (1872, 89), "2022-05-03T16:40:00", ["fc"])),
    (fc4b, Expected(1, (1872, 89), "2022-05-03T18:50:00", ["fc"])),
    (zfc4a, Expected(1, (1894, 89), "2022-05-03T15:22:00", ["zfc"])),
    (zfc4b, Expected(1, (1893, 89), "2022-05-03T17:44:00", ["zfc"])),
    (
        dataset4,
        Expected(
            5, (7761, 89), "2022-05-03T15:22:00", ["zfc", "fc", "zfc", "fc", "mvsh"]
        ),
    ),
    (pd_std1, Expected(0, (445, 89), "2023-03-31T19:25:00", ["mvsh"])),
]


@pytest.mark.parametrize("dat_file,expected", parameterized)
class TestDatFile:
    def test_experiment_type(self, dat_file: DatFile, expected: Expected):
        assert dat_file.experiment_type == expected.experiment

    def test_num_comments(self, dat_file: DatFile, expected: Expected):
        assert len(dat_file.comments) == expected.num_comments

    def test_data_shape(self, dat_file: DatFile, expected: Expected):
        assert dat_file.data.shape == expected.shape

    def test_date_created(self, dat_file: DatFile, expected: Expected):
        assert dat_file.date_created.isoformat() == expected.date

    def test_experiments(self, dat_file: DatFile, expected: Expected):
        assert dat_file.experiments_in_file == expected.exps


expected_comments = [
    (mvsh4, OrderedDict([(0, ["MvsH", "293"])])),
    (mvsh5, OrderedDict([(0, ["MvsH", "20 C"])])),
    (
        zfcfc4,
        OrderedDict(
            [
                (0, ["ZFC", "100"]),
                (1894, ["FC", "100"]),
                (3766, ["ZFC", "1000"]),
                (5659, ["FC", "1000"]),
            ]
        ),
    ),
    (fc4a, OrderedDict([(0, ["FC", "100"])])),
    (fc4b, OrderedDict([(0, ["FC", "1000"])])),
    (zfc4a, OrderedDict([(0, ["ZFC", "100"])])),
    (zfc4b, OrderedDict([(0, ["ZFC", "1000"])])),
    (
        dataset4,
        OrderedDict(
            [
                (0, ["ZFC", "100"]),
                (1894, ["FC", "100"]),
                (3766, ["ZFC", "1000"]),
                (5659, ["FC", "1000"]),
                (7531, ["MvsH", "293"]),
            ]
        ),
    ),
]


@pytest.mark.parametrize("dat_file,expected", expected_comments)
def test_expected_comments(dat_file: DatFile, expected: OrderedDict[int, list[str]]):
    assert dat_file.comments == expected
