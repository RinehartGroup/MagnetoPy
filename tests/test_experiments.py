from dataclasses import dataclass
import inspect
from pathlib import Path

import pytest

from magnetopy import DatFile, MvsH
from magnetopy.experiments import FC, ZFC, ZFCFC, _num_digits_after_decimal

TESTS_PATH = Path(inspect.getfile(inspect.currentframe())).parent
DATA_PATH = TESTS_PATH / "data"

mvsh1_path = DATA_PATH / "mvsh1.dat"
mvsh2_path = DATA_PATH / "mvsh2.dat"
mvsh2a_path = DATA_PATH / "mvsh2a.dat"
mvsh2b_path = DATA_PATH / "mvsh2b.dat"
mvsh3_path = DATA_PATH / "mvsh3.dat"
mvsh4_path = DATA_PATH / "mvsh4.dat"
mvsh5_path = DATA_PATH / "mvsh5.dat"
mvsh5rw_path = DATA_PATH / "mvsh5.rw.dat"
mvsh6_path = DATA_PATH / "mvsh6.dat"
mvsh7_path = DATA_PATH / "mvsh7.dat"
mvsh8_path = DATA_PATH / "mvsh8.dat"
mvsh9_path = DATA_PATH / "mvsh9.dat"
mvsh10_path = DATA_PATH / "mvsh10.dat"
mvsh11_path = DATA_PATH / "mvsh11.dat"
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
pd_std1_path = DATA_PATH / "Pd_std1.dat"

mvsh1_dat = DatFile(mvsh1_path)


@pytest.mark.parametrize(
    "num,digits", [(0, 0), (0.5, 1), (1, 0), (1.25, 2), (2.123, 3), (-1, 0), (-1.2, 1)]
)
def test_num_digits_after_decimal(num, digits):
    assert _num_digits_after_decimal(num) == digits


### MvsH tests ###


@dataclass
class _ExpectedMvsH:
    temperature: int | float
    start_field: int  # manually inspected field value in Oe
    end_field: int  # manually inspected field value in Oe


uncommented_mvsh_expected = [
    ((mvsh1_path, 2), _ExpectedMvsH(2, 70000, 70000)),
    ((mvsh1_dat, 2), _ExpectedMvsH(2, 70000, 70000)),
    ((mvsh1_path, 4), _ExpectedMvsH(4, 70000, 70000)),
    ((mvsh1_path, 6), _ExpectedMvsH(6, 70000, 70000)),
    ((mvsh1_path, 8), _ExpectedMvsH(8, 70000, 70000)),
    ((mvsh1_path, 10), _ExpectedMvsH(10, 70000, 70000)),
    ((mvsh1_path, 12), _ExpectedMvsH(12, 70000, 70000)),
    ((mvsh1_path, 300), _ExpectedMvsH(300, 70000, -70000)),
    ((mvsh2_path, 5), _ExpectedMvsH(5, -70000, -70000)),
    ((mvsh2_path, 300), _ExpectedMvsH(300, -70000, -70000)),
    ((mvsh2a_path, 5), _ExpectedMvsH(5, -70000, -70000)),
    ((mvsh2b_path, 300), _ExpectedMvsH(300, -70000, -70000)),
    ((mvsh3_path, 5), _ExpectedMvsH(5, -70000, -70000)),
    ((mvsh6_path, 300), _ExpectedMvsH(300, -70000, -70000)),
    ((mvsh6_path, 300.0), _ExpectedMvsH(300.0, -70000, -70000)),
    ((mvsh7_path, 300), _ExpectedMvsH(300, -70000, -70000)),
    ((mvsh7_path, 300.1), _ExpectedMvsH(300.1, -70000, -70000)),
    ((mvsh8_path, 2), _ExpectedMvsH(2, 0, 70000)),
    ((mvsh8_path, 2.0), _ExpectedMvsH(2.0, 0, 70000)),
    ((mvsh9_path, 2), _ExpectedMvsH(2, 0, 70000)),
    ((mvsh10_path, 5), _ExpectedMvsH(5, -70000, -70000)),
    ((mvsh11_path, 5), _ExpectedMvsH(5, 0, 70000)),
    ((pd_std1_path, 300), _ExpectedMvsH(300, -70000, -70000)),
    ((pd_std1_path, 300.1), _ExpectedMvsH(300.1, -70000, -70000)),
]


@pytest.mark.parametrize("args,expected", uncommented_mvsh_expected)
class TestUncommentedMvsh:
    def test_temperature(
        self, args: tuple[DatFile, int | float], expected: _ExpectedMvsH
    ):
        mvsh = MvsH(dat_file=args[0], temperature=args[1])
        assert mvsh.temperature == expected.temperature

    def test_endpoint_fields(
        self, args: tuple[DatFile, int | float], expected: _ExpectedMvsH
    ):
        mvsh = MvsH(dat_file=args[0], temperature=args[1])
        found_start_field = round(mvsh.data["Magnetic Field (Oe)"][0])
        found_end_field = round(mvsh.data["Magnetic Field (Oe)"].iloc[-1])
        assert found_start_field == expected.start_field
        assert found_end_field == expected.end_field


commented_mvsh_expected = [
    ((mvsh4_path, 293), _ExpectedMvsH(293, -70000, -70000)),
    ((mvsh5_path, 293), _ExpectedMvsH(293, -70000, -70000)),
    ((dataset4_path, 293), _ExpectedMvsH(293, -70000, -70000)),
]


@pytest.mark.parametrize("args,expected", commented_mvsh_expected)
class TestCommentedMvsh:
    def test_temperature(
        self, args: tuple[DatFile, int | float], expected: _ExpectedMvsH
    ):
        mvsh = MvsH(dat_file=args[0], temperature=args[1])
        assert mvsh.temperature == expected.temperature

    def test_endpoint_fields(
        self, args: tuple[DatFile, int | float], expected: _ExpectedMvsH
    ):
        mvsh = MvsH(dat_file=args[0], temperature=args[1])
        found_start_field = round(mvsh.data["Magnetic Field (Oe)"][0])
        found_end_field = round(mvsh.data["Magnetic Field (Oe)"].iloc[-1])
        assert found_start_field == expected.start_field
        assert found_end_field == expected.end_field


autodetected_temperature_expected = [
    ((DatFile(mvsh2a_path), None), 5),
    ((DatFile(mvsh2b_path), None), 300),
    ((DatFile(mvsh3_path), None), 5),
    ((DatFile(mvsh4_path), None), 293),
    ((DatFile(dataset4_path), None), 293),
    ((DatFile(mvsh5_path), None), 293),
    ((DatFile(mvsh6_path), None), 300),
    ((DatFile(mvsh7_path), None), 300),
    ((DatFile(mvsh7_path), 1), 300.1),
    ((DatFile(mvsh8_path), None), 2),
    ((DatFile(mvsh9_path), None), 2),
    ((DatFile(mvsh10_path), None), 5),
    ((DatFile(mvsh11_path), None), 5),
    ((DatFile(pd_std1_path), None), 300),
    ((DatFile(pd_std1_path), 1), 300.1),
]


@pytest.mark.parametrize("args,expected", autodetected_temperature_expected)
def test_mvsh_auto_temperature_detection(
    args: tuple[DatFile, int | None], expected: int | float
):
    eps = 0.001
    min_samples = 10
    if args[1] is None:
        assert (
            MvsH._auto_detect_temperature(  # pylint: disable=protected-access
                args[0], eps, min_samples, 0
            )
            == expected
        )
    else:
        assert (
            MvsH._auto_detect_temperature(  # pylint: disable=protected-access
                args[0], eps, min_samples, ndigits=args[1]
            )
            == expected
        )


files_w_multiple_temperatures = [
    DatFile(mvsh1_path),
    DatFile(mvsh2_path),
]


@pytest.mark.parametrize("dat_file", files_w_multiple_temperatures)
def test_mvsh_multiple_temperatures(dat_file: DatFile):
    with pytest.raises(MvsH.AutoReadError):
        MvsH(dat_file)


@dataclass
class _ExpectedSegment:
    # each tuple is the rounded starting and ending field values
    # tuple[start, end] where start and end are integers in Oe
    virgin: tuple[int, int] | None
    reverse: tuple[int, int] | None
    forward: tuple[int, int] | None


mvsh_segments_expected = [
    (MvsH(mvsh1_path, 2), _ExpectedSegment(None, (70000, -70000), (-70000, 70000))),
    (MvsH(mvsh1_path, 4), _ExpectedSegment(None, (70000, -70000), (-70000, 70000))),
    (MvsH(mvsh1_path, 6), _ExpectedSegment(None, (70000, -70000), (-70000, 70000))),
    (MvsH(mvsh1_path, 8), _ExpectedSegment(None, (70000, -70000), (-70000, 70000))),
    (MvsH(mvsh1_path, 10), _ExpectedSegment(None, (70000, -70000), (-70000, 70000))),
    (MvsH(mvsh1_path, 12), _ExpectedSegment(None, (70000, -70000), (-70000, 70000))),
    (MvsH(mvsh1_path, 300), _ExpectedSegment(None, (70000, -70000), None)),
    (MvsH(mvsh2_path, 5), _ExpectedSegment(None, (70000, -70000), (-70000, 70000))),
    (MvsH(mvsh2_path, 300), _ExpectedSegment(None, (70000, -70000), (-70000, 70000))),
    (MvsH(mvsh3_path, 5), _ExpectedSegment(None, (70000, -70000), (-70000, 70000))),
    (MvsH(mvsh4_path, 293), _ExpectedSegment(None, (70000, -70000), (-70000, 70000))),
    (MvsH(mvsh5_path, 293), _ExpectedSegment(None, (70000, -70000), (-70000, 70000))),
    (MvsH(mvsh6_path, 300), _ExpectedSegment(None, (70000, -70000), (-70000, 70000))),
    (MvsH(mvsh7_path, 300), _ExpectedSegment(None, (70000, -70000), (-70000, 70000))),
    (
        MvsH(mvsh8_path, 2),
        _ExpectedSegment((0, 70039), (70039, -70026), (-70026, 70000)),
    ),
    (
        MvsH(mvsh9_path, 2),
        _ExpectedSegment((0, 70000), (70000, -70000), (-70000, 70000)),
    ),
    (MvsH(mvsh10_path, 5), _ExpectedSegment(None, (70000, -70000), (-70000, 70000))),
    (
        MvsH(mvsh11_path, 5),
        _ExpectedSegment((0, 70000), (70000, -70000), (-70000, 70000)),
    ),
    (MvsH(pd_std1_path, 300), _ExpectedSegment(None, (70000, -70000), (-70000, 70000))),
]


@pytest.mark.parametrize("args,expected", mvsh_segments_expected)
class TestMvsHSegments:
    def test_virgin(self, args: MvsH, expected: _ExpectedSegment):
        if expected.virgin is None:
            with pytest.raises(MvsH.SegmentError):
                args.virgin  # pylint: disable=pointless-statement
        else:
            found_start_field = round(args.virgin["Magnetic Field (Oe)"][0])
            found_end_field = round(args.virgin["Magnetic Field (Oe)"].iloc[-1])
            assert found_start_field == expected.virgin[0]
            assert found_end_field == expected.virgin[1]

    def test_reverse(self, args: MvsH, expected: _ExpectedSegment):
        if expected.reverse is None:
            with pytest.raises(MvsH.SegmentError):
                args.reverse  # pylint: disable=pointless-statement
        else:
            found_start_field = round(args.reverse["Magnetic Field (Oe)"][0])
            found_end_field = round(args.reverse["Magnetic Field (Oe)"].iloc[-1])
            assert found_start_field == expected.reverse[0]
            assert found_end_field == expected.reverse[1]

    def test_forward(self, args: MvsH, expected: _ExpectedSegment):
        if expected.forward is None:
            with pytest.raises(MvsH.SegmentError):
                args.forward  # pylint: disable=pointless-statement
        else:
            found_start_field = round(args.forward["Magnetic Field (Oe)"][0])
            found_end_field = round(args.forward["Magnetic Field (Oe)"].iloc[-1])
            assert found_start_field == expected.forward[0]
            assert found_end_field == expected.forward[1]


### ZFCFC tests ###

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
    (("filename.dat", "zfc", False), "unknown"),
]


@pytest.mark.parametrize("args,expected", filenames_expected)
def test_zfcfc_filename_label(args, expected):
    assert ZFCFC._filename_label(*args) == expected


filenames_warnings_expected = [
    (("zfc1.dat", "fc", False), "zfc"),
    (("fc1.dat", "zfc", False), "fc"),
]


@pytest.mark.parametrize("args,expected", filenames_warnings_expected)
def test_zfcfc_filename_label_warnings(args, expected):
    with pytest.warns(ZFCFC.FileNameWarning):
        assert ZFCFC._filename_label(*args) == expected


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
def test_zfcfc_autodetect_field_uncommented(args, expected):
    assert ZFCFC._autodetect_field(*args) == expected


fields_commented_expected = [
    ((DatFile(zfc4a_path), "zfc", 0), 100),
    ((DatFile(zfc4b_path), "zfc", 0), 1000),
    ((DatFile(fc4a_path), "fc", 0), 100),
    ((DatFile(fc4b_path), "fc", 0), 1000),
]


@pytest.mark.parametrize("args,expected", fields_commented_expected)
def test_zfcfc_autodetect_field_commented(args, expected):
    assert ZFCFC._autodetect_field(*args) == expected


def test_zfcfc_autodetect_field_error():
    with pytest.raises(ZFCFC.AutoReadError):
        ZFCFC._autodetect_field(DatFile(zfcfc4_path), "zfc", 0)


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


def test_zfc_autoread_error():
    with pytest.raises(ZFC.AutoReadError):
        ZFC(DatFile(zfcfc4_path))


def test_fc_autoread_error():
    with pytest.raises(FC.AutoReadError):
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
    with pytest.warns(cls.FileNameWarning):
        cls(path)
