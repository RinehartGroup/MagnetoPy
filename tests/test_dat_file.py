import inspect
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import shutil
import numpy as np

import pytest

from magnetopy import DatFile
from magnetopy.data_files import DcMeasurement, filename_label, FileNameWarning

filename_expected = [
    ("mvsh1.dat", "", "mvsh"),
    ("zfcfc1.dat", "", "zfcfc"),
    ("zfc1.dat", "", "zfc"),
    ("fc1.dat", "", "fc"),
    ("zfcfc_zfc.dat", "", "zfcfc"),
    ("dataset", "", "unknown"),
]


@pytest.mark.parametrize("filename,experiment,expected", filename_expected)
def test_filename_label(filename: str, experiment: str, expected: str):
    assert filename_label(filename, experiment, False) == expected


filename_expected_warnings = [
    ("zfc1.dat", "fc", "zfc"),
    ("fc1.dat", "zfc", "fc"),
]


@pytest.mark.parametrize("filename,experiment,expected", filename_expected_warnings)
def test_filename_label_warning(filename: str, experiment: str, expected: str):
    with pytest.warns(FileNameWarning):
        assert filename_label(filename, experiment, False) == expected


TESTS_PATH = Path(inspect.getfile(inspect.currentframe())).parent
DATA_PATH = TESTS_PATH / "data"

mvsh1 = DATA_PATH / "mvsh1.dat"
mvsh2 = DATA_PATH / "mvsh2.dat"
mvsh2a = DATA_PATH / "mvsh2a.dat"
mvsh2b = DATA_PATH / "mvsh2b.dat"
mvsh3 = DATA_PATH / "mvsh3.dat"
mvsh4 = DATA_PATH / "mvsh4.dat"
mvsh5 = DATA_PATH / "mvsh5.dat"
mvsh5rw = DATA_PATH / "mvsh5.rw.dat"
mvsh6 = DATA_PATH / "mvsh6.dat"
mvsh7 = DATA_PATH / "mvsh7.dat"
mvsh8 = DATA_PATH / "mvsh8.dat"
mvsh9 = DATA_PATH / "mvsh9.dat"
mvsh10 = DATA_PATH / "mvsh10.dat"
mvsh11 = DATA_PATH / "mvsh11.dat"
zfcfc1 = DATA_PATH / "zfcfc1.dat"
zfcfc2 = DATA_PATH / "zfcfc2.dat"
zfcfc3 = DATA_PATH / "zfcfc3.dat"
zfcfc4 = DATA_PATH / "zfcfc4.dat"
fc4a = DATA_PATH / "fc4a.dat"
fc4b = DATA_PATH / "fc4b.dat"
fc5 = DATA_PATH / "fc5.dat"
fc5rw = DATA_PATH / "fc5.rw.dat"
zfc4a = DATA_PATH / "zfc4a.dat"
zfc4b = DATA_PATH / "zfc4b.dat"
zfc5 = DATA_PATH / "zfc5.dat"
zfcrw = DATA_PATH / "zfc5.rw.dat"
dataset4 = DATA_PATH / "dataset4.dat"
pd_std1 = DATA_PATH / "Pd_std1.dat"


class TestDatFileBaseAttrs:
    """This is essentially a repeat of the GenericFile test and will only look at
    attributes created by the GenericFile base class. Note that since DatFile
    overwrites the date_created attribute, it won't be included in this test.
    """

    @pytest.fixture(scope="class")
    def mvsh1_dat_file(self):
        return DatFile(mvsh1)

    def test_dat_file_repr(self, mvsh1_dat_file: DatFile):
        assert repr(mvsh1_dat_file) == "DatFile(mvsh1.dat)"

    def test_dat_file_str(self, mvsh1_dat_file: DatFile):
        assert str(mvsh1_dat_file) == "DatFile(mvsh1.dat)"

    def test_dat_file_exp_type(self, mvsh1_dat_file: DatFile):
        assert mvsh1_dat_file.experiment_type == "magnetometry"

    def test_dat_file_local_path(self, mvsh1_dat_file: DatFile):
        assert mvsh1_dat_file.local_path == Path(DATA_PATH / "mvsh1.dat")

    def test_dat_file_as_dict(self, mvsh1_dat_file: DatFile):
        serialized = mvsh1_dat_file.as_dict()
        assert serialized["local_path"] == str(Path(DATA_PATH / "mvsh1.dat"))


@dataclass
class _Expected:
    experiment = "magnetometry"
    num_comments: int
    shape: tuple[int, int]
    date: str
    exps: list[str]


parameterized = [
    (mvsh1, _Expected(0, (7305, 89), "2020-07-11T11:07:00", ["mvsh"])),
    (mvsh2, _Expected(0, (719, 89), "2022-03-25T14:11:00", ["mvsh"])),
    (mvsh2a, _Expected(0, (274, 89), "2022-03-25T14:11:00", ["mvsh"])),
    (mvsh2b, _Expected(0, (445, 89), "2022-03-25T14:11:00", ["mvsh"])),
    (mvsh3, _Expected(0, (141, 89), "2019-09-21T01:57:00", ["mvsh"])),
    (mvsh4, _Expected(1, (230, 89), "2022-05-03T22:44:00", ["mvsh"])),
    (mvsh5, _Expected(1, (230, 89), "2022-10-10T23:44:00", ["mvsh"])),
    (mvsh5rw, _Expected(458, (138316, 7), "2022-10-11T00:00:00", [])),
    (mvsh6, _Expected(0, (445, 89), "2021-10-02T21:00:00", ["mvsh"])),
    (mvsh7, _Expected(0, (445, 89), "2022-08-24T08:08:00", ["mvsh"])),
    (mvsh8, _Expected(0, (7872, 89), "2022-11-01T19:47:00", ["mvsh"])),
    (mvsh9, _Expected(0, (425, 89), "2021-11-28T05:58:00", ["mvsh"])),
    (mvsh10, _Expected(0, (285, 89), "2023-03-01T19:47:00", ["mvsh"])),
    (mvsh11, _Expected(0, (3407, 89), "2023-03-02T05:49:00", ["mvsh"])),
    (zfcfc1, _Expected(0, (504, 89), "2021-09-18T19:46:00", ["zfcfc"])),
    (zfcfc2, _Expected(0, (494, 89), "2021-09-06T23:20:00", ["zfcfc"])),
    (zfcfc3, _Expected(0, (513, 89), "2022-11-10T17:19:00", ["zfcfc"])),
    (
        zfcfc4,
        _Expected(4, (7531, 89), "2022-05-03T15:22:00", ["zfc", "fc", "zfc", "fc"]),
    ),
    (fc4a, _Expected(1, (1872, 89), "2022-05-03T16:40:00", ["fc"])),
    (fc4b, _Expected(1, (1872, 89), "2022-05-03T18:50:00", ["fc"])),
    (zfc4a, _Expected(1, (1894, 89), "2022-05-03T15:22:00", ["zfc"])),
    (zfc4b, _Expected(1, (1893, 89), "2022-05-03T17:44:00", ["zfc"])),
    (
        dataset4,
        _Expected(
            5, (7761, 89), "2022-05-03T15:22:00", ["zfc", "fc", "zfc", "fc", "mvsh"]
        ),
    ),
    (pd_std1, _Expected(0, (445, 89), "2023-03-31T19:25:00", ["mvsh"])),
]


@pytest.mark.parametrize("dat_file_path,expected", parameterized)
class TestDatFile:
    def test_experiment_type(self, dat_file_path: Path, expected: _Expected):
        assert DatFile(dat_file_path).experiment_type == expected.experiment

    def test_num_comments(self, dat_file_path: Path, expected: _Expected):
        assert len(DatFile(dat_file_path).comments) == expected.num_comments

    def test_data_shape(self, dat_file_path: Path, expected: _Expected):
        assert DatFile(dat_file_path).data.shape == expected.shape

    def test_date_created(self, dat_file_path: Path, expected: _Expected):
        assert DatFile(dat_file_path).date_created.isoformat() == expected.date

    def test_experiments(self, dat_file_path: Path, expected: _Expected):
        assert DatFile(dat_file_path).experiments_in_file == expected.exps

    def test_experiments_from_ambiguosly_named_files(
        self, tmp_path, dat_file_path: Path, expected: _Expected
    ):
        temp_file = tmp_path / "temp.dat"
        shutil.copy(dat_file_path, temp_file)
        assert DatFile(temp_file).experiments_in_file == expected.exps


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


@pytest.mark.parametrize("dat_file_path,expected", expected_comments)
def test_expected_comments(dat_file_path: Path, expected: OrderedDict[int, list[str]]):
    assert DatFile(dat_file_path).comments == expected


### Test .rw.dat functionality ###


def test_empty_comment_row():
    """Rows with comments should have empty 'raw_scan' rows."""
    mvsh5raw = DatFile(mvsh5, True)
    assert np.isnan(mvsh5raw.data["raw_scan"].iloc[0])


@dataclass
class _ExpectedTimes:
    up_start: float
    up_end: float
    down_start: float
    down_end: float
    processed_start: float
    processed_end: float


expected_times = [
    (
        mvsh5,
        1,
        _ExpectedTimes(
            3874607983.80602,
            3874607987.80602,
            3874607988.52663,
            3874607992.52663,
            3874607993.10447,
            3874607997.08447,
        ),
    ),
    (
        mvsh5,
        -1,
        _ExpectedTimes(
            3874613954.71453,
            3874613958.71453,
            3874613959.41793,
            3874613963.41793,
            3874613964.07055,
            3874613968.05055,
        ),
    ),
    (
        zfc5,
        0,
        _ExpectedTimes(
            3876038306.03405,
            3876038310.03405,
            3876038310.75794,
            3876038314.75794,
            3876038315.32042,
            3876038319.30042,
        ),
    ),
    (
        zfc5,
        -1,
        _ExpectedTimes(
            3876059006.14798,
            3876059010.14798,
            3876059010.92101,
            3876059014.92101,
            3876059015.52881,
            3876059019.50881,
        ),
    ),
    (
        fc5,
        0,
        _ExpectedTimes(
            3876059853.86451,
            3876059857.86451,
            3876059858.57388,
            3876059862.57388,
            3876059863.15624,
            3876059867.13624,
        ),
    ),
    (
        fc5,
        -1,
        _ExpectedTimes(
            3876080563.16431,
            3876080567.16431,
            3876080567.89845,
            3876080571.89845,
            3876080572.48217,
            3876080576.46217,
        ),
    ),
]


@pytest.mark.parametrize("dat_file_path,idx,expected", expected_times)
def test_expected_times(dat_file_path: Path, idx: int, expected: _ExpectedTimes):
    """
    This test just checks that recorded times in the .rw.dat file in the the first and
    last data points in an experiment are correct.
    """
    dat_file = DatFile(dat_file_path, True)
    meas: DcMeasurement = dat_file.data["raw_scan"].iloc[idx]
    assert meas.up.data["Time Stamp (sec)"].iloc[0] == expected.up_start
    assert meas.up.data["Time Stamp (sec)"].iloc[-1] == expected.up_end
    assert meas.down.data["Time Stamp (sec)"].iloc[0] == expected.down_start
    assert meas.down.data["Time Stamp (sec)"].iloc[-1] == expected.down_end
    assert (
        meas.processed_scan.data["Time Stamp (sec)"].iloc[0] == expected.processed_start
    )
    assert (
        meas.processed_scan.data["Time Stamp (sec)"].iloc[-1] == expected.processed_end
    )
