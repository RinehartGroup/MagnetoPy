from dataclasses import dataclass
import inspect
from pathlib import Path

import pytest

from magnetopy.data_files import DatFile, FileNameWarning
from magnetopy.experiments import (
    FC,
    ZFC,
    ZFCFC,
    filename_label,
    _auto_detect_field,
    FieldDetectionError,
)

TESTS_PATH = Path(inspect.getfile(inspect.currentframe())).parent
DATA_PATH = TESTS_PATH / "data"

zfcfc1_path = DATA_PATH / "zfcfc1.dat"
zfcfc2_path = DATA_PATH / "zfcfc2.dat"
zfcfc3_path = DATA_PATH / "zfcfc3.dat"
zfcfc4_path = DATA_PATH / "zfcfc4.dat"
fc4a_path = DATA_PATH / "fc4a.dat"
fc4b_path = DATA_PATH / "fc4b.dat"
zfc4a_path = DATA_PATH / "zfc4a.dat"
zfc4b_path = DATA_PATH / "zfc4b.dat"
fc5_path = DATA_PATH / "fc5.dat"
zfc5_path = DATA_PATH / "zfc5.dat"
dataset4_path = DATA_PATH / "dataset4.dat"

filenames_expected = [
    (("zfcfc1.dat", "zfc", False), "zfcfc"),
    (("zfcfc1.dat", "fc", False), "zfcfc"),
    (("zfc1.dat", "zfc", False), "zfc"),
    (("fc1.dat", "fc", False), "fc"),
    (("zfcfc1.dat", "zfc", True), "zfcfc"),
    (("zfcfc1.dat", "fc", True), "zfcfc"),
    (("zfc1.dat", "zfc", True), "zfc"),
    (("zfc1.dat", "fc", True), "zfc"),
    (("fc1.dat", "zfc", True), "fc"),
    (("fc1.dat", "fc", True), "fc"),
    (("zfcfc_zfc.dat", "zfc", False), "zfcfc"),
    (("zfcfc_zfc.dat", "fc", False), "zfcfc"),
    (("mvsh1.dat", "", False), "mvsh"),
    (("filename.dat", "zfc", False), "unknown"),
]


@pytest.mark.parametrize("args,expected", filenames_expected)
def test_zfcfc_filename_label(args, expected):
    assert filename_label(*args) == expected


filenames_warnings_expected = [
    (("zfc1.dat", "fc", False), "zfc"),
    (("fc1.dat", "zfc", False), "fc"),
]


@pytest.mark.parametrize("args,expected", filenames_warnings_expected)
def test_zfcfc_filename_label_warnings(args, expected):
    with pytest.warns(FileNameWarning):
        assert filename_label(*args) == expected


fields_uncommented_expected = [
    ((DatFile(zfcfc1_path), "zfc", 0), 100),
    ((DatFile(zfcfc2_path), "zfc", 0), 100),
    ((DatFile(zfcfc3_path), "zfc", 0), 100),
    ((DatFile(zfcfc1_path), "zfc", 1), 100.1),
    ((DatFile(zfcfc2_path), "zfc", 1), 99.9),
    ((DatFile(zfcfc3_path), "zfc", 1), 100.0),
    ((DatFile(zfcfc1_path), "fc", 0), 100),
    ((DatFile(zfcfc2_path), "fc", 0), 100),
    ((DatFile(zfcfc3_path), "fc", 0), 100),
    ((DatFile(zfcfc1_path), "fc", 1), 100.1),
    ((DatFile(zfcfc2_path), "fc", 1), 99.9),
    ((DatFile(zfcfc3_path), "fc", 1), 100.0),
    ((DatFile(zfc5_path), "zfc", 0), 200),
    ((DatFile(fc5_path), "fc", 0), 200),
]


@pytest.mark.parametrize("args,expected", fields_uncommented_expected)
def test_zfcfc_auto_detect_field_uncommented(args, expected):
    assert _auto_detect_field(*args) == expected


fields_commented_expected = [
    ((DatFile(zfc4a_path), "zfc", 0), 100),
    ((DatFile(zfc4b_path), "zfc", 0), 1000),
    ((DatFile(fc4a_path), "fc", 0), 100),
    ((DatFile(fc4b_path), "fc", 0), 1000),
]


@pytest.mark.parametrize("args,expected", fields_commented_expected)
def test_zfcfc_auto_detect_field_commented(args, expected):
    assert _auto_detect_field(*args) == expected


def test_zfcfc_autodetect_field_error():
    with pytest.raises(FieldDetectionError):
        _auto_detect_field(DatFile(zfcfc4_path), "zfc", 0)


@dataclass
class _ExpectedZFCFC:
    field: int | float
    start_temp: int | float
    end_temp: int | float


expected_zfc = [
    (zfcfc1_path, _ExpectedZFCFC(100, 5.00255632400513, 299.939453125)),
    (zfcfc2_path, _ExpectedZFCFC(100, 5.0022759437561, 339.943023681641)),
    (zfcfc3_path, _ExpectedZFCFC(100, 5.00209665298462, 299.907318115234)),
    (zfc5_path, _ExpectedZFCFC(200, 2.00167036056519, 299.911483764648)),
]


@pytest.mark.parametrize("path,expected", expected_zfc)
def test_zfc(path: Path, expected: _ExpectedZFCFC):
    zfc = ZFC(path)
    zfc_start = zfc.data["Temperature (K)"].iloc[0]
    zfc_end = zfc.data["Temperature (K)"].iloc[-1]
    assert zfc.field == expected.field
    assert zfc_start == expected.start_temp
    assert zfc_end == expected.end_temp


expected_fc = [
    (zfcfc1_path, _ExpectedZFCFC(100, 5.00277781486511, 299.924865722656)),
    (zfcfc2_path, _ExpectedZFCFC(100, 5.00186157226562, 339.937866210937)),
    (zfcfc3_path, _ExpectedZFCFC(100, 299.927917480469, 5.00218033790588)),
    (fc5_path, _ExpectedZFCFC(200, 2.00117123126984, 299.906265258789)),
]


@pytest.mark.parametrize("path,expected", expected_fc)
def test_fc(path: Path, expected: _ExpectedZFCFC):
    fc = FC(path)
    fc_start = fc.data["Temperature (K)"].iloc[0]
    fc_end = fc.data["Temperature (K)"].iloc[-1]
    assert fc.field == expected.field
    assert fc_start == expected.start_temp
    assert fc_end == expected.end_temp


def test_zfc_field_detection_error():
    with pytest.raises(FieldDetectionError):
        ZFC(DatFile(zfcfc4_path))


def test_fc_field_detection_error():
    with pytest.raises(FieldDetectionError):
        FC(DatFile(zfcfc4_path))


expected_zfc_requiring_field = [
    ((zfcfc4_path, 100), _ExpectedZFCFC(100, 4.999943733, 309.9239655)),
    ((zfcfc4_path, 1000), _ExpectedZFCFC(1000, 4.999952793, 309.9181976)),
    ((dataset4_path, 100), _ExpectedZFCFC(100, 4.999943733, 309.9239655)),
    ((dataset4_path, 1000), _ExpectedZFCFC(1000, 4.999952793, 309.9181976)),
]


@pytest.mark.parametrize("args,expected", expected_zfc_requiring_field)
def test_zfc_requiring_field(args: tuple[Path, int], expected: _ExpectedZFCFC):
    zfc = ZFC(*args)
    zfc_start = zfc.data["Temperature (K)"].iloc[0]
    zfc_end = zfc.data["Temperature (K)"].iloc[-1]
    assert zfc.field == expected.field
    assert zfc_start == expected.start_temp
    assert zfc_end == expected.end_temp


expected_fc_requiring_field = [
    ((zfcfc4_path, 100), _ExpectedZFCFC(100, 309.9851379, 4.999839783)),
    ((zfcfc4_path, 1000), _ExpectedZFCFC(1000, 310.0117493, 4.999486685)),
    ((dataset4_path, 100), _ExpectedZFCFC(100, 309.9851379, 4.999839783)),
    ((dataset4_path, 1000), _ExpectedZFCFC(1000, 310.0117493, 4.999486685)),
]


@pytest.mark.parametrize("args,expected", expected_fc_requiring_field)
def test_fc_requiring_field(args: tuple[Path, int], expected: _ExpectedZFCFC):
    fc = FC(*args)
    fc_start = fc.data["Temperature (K)"].iloc[0]
    fc_end = fc.data["Temperature (K)"].iloc[-1]
    assert fc.field == expected.field
    assert fc_start == expected.start_temp
    assert fc_end == expected.end_temp


expected_filename_warning = [(ZFC, fc5_path), (FC, zfc5_path)]


@pytest.mark.parametrize("cls,path", expected_filename_warning)
def test_zfcfc_filename_warning(cls: ZFCFC, path):
    with pytest.warns(FileNameWarning):
        cls(path)


class TestGetAllZFCFCInFile:
    def test_single_zfcfc_in_file(self):
        zfc_objs = ZFC.get_all_in_file(zfcfc1_path)
        fc_objs = FC.get_all_in_file(zfcfc1_path)
        assert len(zfc_objs) == len(fc_objs) == 1
        assert zfc_objs[0].field == fc_objs[0].field == 100

    def test_single_zfcfc_in_file_equivalency(self):
        zfc_objs = ZFC.get_all_in_file(zfcfc1_path)
        fc_objs = FC.get_all_in_file(zfcfc1_path)
        zfc = ZFC(zfcfc1_path)
        fc = FC(zfcfc1_path)
        assert (
            zfc_objs[0]
            .data["uncorrected_moment"]
            .equals(zfc.data["uncorrected_moment"])
        )
        assert (
            fc_objs[0].data["uncorrected_moment"].equals(fc.data["uncorrected_moment"])
        )

    def test_multiple_zfcfc_in_file(self):
        zfc_objs = ZFC.get_all_in_file(zfcfc4_path)
        fc_objs = FC.get_all_in_file(zfcfc4_path)
        assert len(zfc_objs) == len(fc_objs) == 2
        assert zfc_objs[0].field == fc_objs[0].field == 100
        assert zfc_objs[1].field == fc_objs[1].field == 1000

    def test_multiple_zfcfc_in_file_equivalency(self):
        zfc_objs = ZFC.get_all_in_file(zfcfc4_path)
        fc_objs = FC.get_all_in_file(zfcfc4_path)
        zfc_100 = ZFC(zfcfc4_path, 100)
        fc_100 = FC(zfcfc4_path, 100)
        zfc_1000 = ZFC(zfcfc4_path, 1000)
        fc_1000 = FC(zfcfc4_path, 1000)
        assert (
            zfc_objs[0]
            .data["uncorrected_moment"]
            .equals(zfc_100.data["uncorrected_moment"])
        )
        assert (
            fc_objs[0]
            .data["uncorrected_moment"]
            .equals(fc_100.data["uncorrected_moment"])
        )
        assert (
            zfc_objs[1]
            .data["uncorrected_moment"]
            .equals(zfc_1000.data["uncorrected_moment"])
        )
        assert (
            fc_objs[1]
            .data["uncorrected_moment"]
            .equals(fc_1000.data["uncorrected_moment"])
        )
