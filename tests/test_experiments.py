from dataclasses import dataclass
import inspect
from pathlib import Path
import numpy as np
import pandas as pd

import pytest

from magnetopy.data_files import DatFile
from magnetopy.experiments import (
    MvsH,
    FC,
    ZFC,
    ZFCFC,
    _num_digits_after_decimal,
    _filename_label,
    _auto_detect_temperature,
    _auto_detect_field,
    _add_uncorrected_moment_columns,
    _scale_dc_data,
    _scale_magnetic_data_mass,
    _scale_magnetic_data_molar_w_eicosane_and_diamagnet,
    FileNameWarning,
    FieldDetectionError,
    TemperatureDetectionError,
)

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


auto_detected_temperature_expected = [
    ((DatFile(mvsh2a_path), 0), 5),
    ((DatFile(mvsh2b_path), 0), 300),
    ((DatFile(mvsh3_path), 0), 5),
    ((DatFile(mvsh4_path), 0), 293),
    ((DatFile(dataset4_path), 0), 293),
    ((DatFile(mvsh5_path), 0), 293),
    ((DatFile(mvsh6_path), 0), 300),
    ((DatFile(mvsh7_path), 0), 300),
    ((DatFile(mvsh7_path), 1), 300.1),
    ((DatFile(mvsh8_path), 0), 2),
    ((DatFile(mvsh9_path), 0), 2),
    ((DatFile(mvsh10_path), 0), 5),
    ((DatFile(mvsh11_path), 0), 5),
    ((DatFile(pd_std1_path), 0), 300),
    ((DatFile(pd_std1_path), 1), 300.1),
]


@pytest.mark.parametrize("args,expected", auto_detected_temperature_expected)
def test_mvsh_auto_temperature_detection(
    args: tuple[DatFile, int | None], expected: int | float
):
    dat_file = args[0]
    eps = 0.001
    min_samples = 10
    n_digits = args[1]
    assert _auto_detect_temperature(dat_file, eps, min_samples, n_digits) == expected


files_w_multiple_temperatures = [
    DatFile(mvsh1_path),
    DatFile(mvsh2_path),
]


@pytest.mark.parametrize("dat_file", files_w_multiple_temperatures)
def test_mvsh_multiple_temperatures(dat_file: DatFile):
    with pytest.raises(TemperatureDetectionError):
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
    (("mvsh1.dat", "", False), "mvsh"),
    (("filename.dat", "zfc", False), "unknown"),
]


@pytest.mark.parametrize("args,expected", filenames_expected)
def test_zfcfc_filename_label(args, expected):
    assert _filename_label(*args) == expected


filenames_warnings_expected = [
    (("zfc1.dat", "fc", False), "zfc"),
    (("fc1.dat", "zfc", False), "fc"),
]


@pytest.mark.parametrize("args,expected", filenames_warnings_expected)
def test_zfcfc_filename_label_warnings(args, expected):
    with pytest.warns(FileNameWarning):
        assert _filename_label(*args) == expected


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


### Test Data Scaling ###


class MockDcExperiment:
    def __init__(self):
        self.data = pd.DataFrame(
            {
                "Temperature (K)": [300],
                "Magnetic Field (Oe)": [1000],
                "Moment (emu)": [np.nan],
                "M. Std. Err. (emu)": [np.nan],
                "DC Moment Free Ctr (emu)": [10],
                "DC Moment Err Free Ctr (emu)": [1],
            }
        )
        self.scaling: list[str] = []


class MockVsmExperiment:
    def __init__(self):
        self.data = pd.DataFrame(
            {
                "Temperature (K)": [300],
                "Magnetic Field (Oe)": [1000],
                "Moment (emu)": [10],
                "M. Std. Err. (emu)": [1],
                "DC Moment Free Ctr (emu)": [np.nan],
                "DC Moment Err Free Ctr (emu)": [np.nan],
            }
        )
        self.scaling: list[str] = []


class TestAddUncorrectedMomentColumns:
    def test_dc(self):
        exp = MockDcExperiment()
        _add_uncorrected_moment_columns(exp)
        assert set(["uncorrected_moment", "uncorrected_moment_err"]).issubset(
            set(exp.data.columns)
        )

    def test_vsm(self):
        exp = MockVsmExperiment()
        _add_uncorrected_moment_columns(exp)
        assert set(["uncorrected_moment", "uncorrected_moment_err"]).issubset(
            set(exp.data.columns)
        )


class TestScaleMagneticDataWEicosaneAndDiamagnet:
    @pytest.fixture
    def exp(self):
        exp = MockDcExperiment()
        _add_uncorrected_moment_columns(exp)
        return exp

    def test_molar(self, exp: MockDcExperiment):
        _scale_magnetic_data_molar_w_eicosane_and_diamagnet(exp.data, 1, 0, 0)
        assert exp.data["chi"].iloc[0] == 0.01
        assert exp.data["chi_err"].iloc[0] == 0.001
        assert exp.data["chi_t"].iloc[0] == 3.0
        assert exp.data["chi_t_err"].iloc[0] == 0.3
        assert round(exp.data["moment"].iloc[0], 8) == 0.00179051
        assert round(exp.data["moment_err"].iloc[0], 8) == 0.00017905

    def test_molar_eicosane(self, exp: MockDcExperiment):
        _scale_magnetic_data_molar_w_eicosane_and_diamagnet(exp.data, 1, 1, 0)
        assert round(exp.data["chi"].iloc[0], 8) == 0.01000086
        assert round(exp.data["chi_err"].iloc[0], 8) == 0.00100086
        assert round(exp.data["chi_t"].iloc[0], 8) == 3.00025807
        assert round(exp.data["chi_t_err"].iloc[0], 8) == 0.30025807
        assert round(exp.data["moment"].iloc[0], 8) == 0.00179066
        assert round(exp.data["moment_err"].iloc[0], 8) == 0.00017921

    def test_molar_diamagnet(self, exp: MockDcExperiment):
        _scale_magnetic_data_molar_w_eicosane_and_diamagnet(exp.data, 1, 0, 0.0001)
        assert round(exp.data["chi"].iloc[0], 8) == 0.0099
        assert round(exp.data["chi_err"].iloc[0], 8) == 0.0009
        assert round(exp.data["chi_t"].iloc[0], 8) == 2.97
        assert round(exp.data["chi_t_err"].iloc[0], 8) == 0.27
        assert round(exp.data["moment"].iloc[0], 8) == 0.00177261
        assert round(exp.data["moment_err"].iloc[0], 8) == 0.00016115

    def test_molar_eicosane_diamagnet(self, exp: MockDcExperiment):
        _scale_magnetic_data_molar_w_eicosane_and_diamagnet(exp.data, 1, 1, 0.0001)
        assert round(exp.data["chi"].iloc[0], 8) == 0.00990086
        assert round(exp.data["chi_err"].iloc[0], 8) == 0.00090086
        assert round(exp.data["chi_t"].iloc[0], 8) == 2.97025807
        assert round(exp.data["chi_t_err"].iloc[0], 8) == 0.27025807
        assert round(exp.data["moment"].iloc[0], 8) == 0.00177276
        assert round(exp.data["moment_err"].iloc[0], 8) == 0.0001613


def test_magnetic_data_mass():
    exp = MockVsmExperiment()
    _add_uncorrected_moment_columns(exp)
    _scale_magnetic_data_mass(exp.data, 2)
    assert exp.data["chi"].iloc[0] == 5.0
    assert exp.data["chi_err"].iloc[0] == 0.5
    assert exp.data["chi_t"].iloc[0] == 1500.0
    assert exp.data["chi_t_err"].iloc[0] == 150.0
    assert exp.data["moment"].iloc[0] == 5000.0
    assert exp.data["moment_err"].iloc[0] == 500.0


class TestScaleDcData:
    @pytest.fixture
    def exp(self):
        exp = MockVsmExperiment()
        _add_uncorrected_moment_columns(exp)
        return exp

    def test_none(self, exp: MockVsmExperiment):
        _scale_dc_data(exp)
        assert exp.scaling == []

    def test_mass(self, exp: MockVsmExperiment):
        _scale_dc_data(exp, mass=2)
        assert exp.scaling == ["mass"]

    def test_molar(self, exp: MockVsmExperiment):
        _scale_dc_data(exp, mass=2, molecular_weight=10)
        assert exp.scaling == ["molar"]

    def test_molar_eicosane(self, exp: MockDcExperiment):
        _scale_dc_data(exp, mass=2, eicosane_mass=0.5, molecular_weight=10)
        assert exp.scaling == ["molar", "eicosane"]

    def test_molar_diamagnet(self, exp: MockDcExperiment):
        _scale_dc_data(exp, mass=2, molecular_weight=10, diamagnetic_correction=0.0001)
        assert exp.scaling == ["molar", "diamagnetic_correction"]

    def test_molar_eicosane_diamagnet(self, exp: MockDcExperiment):
        _scale_dc_data(
            exp,
            mass=2,
            eicosane_mass=0.5,
            molecular_weight=10,
            diamagnetic_correction=0.0001,
        )
        assert exp.scaling == ["molar", "eicosane", "diamagnetic_correction"]
