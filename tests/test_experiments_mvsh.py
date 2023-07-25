from dataclasses import dataclass
import inspect
from pathlib import Path

import pytest

from magnetopy.data_files import DatFile
from magnetopy.experiments import (
    MvsH,
    _num_digits_after_decimal,
    _auto_detect_temperature,
    TemperatureDetectionError,
)

TESTS_PATH = Path(inspect.getfile(inspect.currentframe())).parent
DATA_PATH = TESTS_PATH / "data"

mvsh1_path = DATA_PATH / "mvsh1.dat"
mvsh2_path = DATA_PATH / "mvsh2.dat"
mvsh2a_path = DATA_PATH / "mvsh2a.dat"
mvsh2b_path = DATA_PATH / "mvsh2b.dat"
mvsh2c_path = DATA_PATH / "mvsh2c.dat"
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
dataset4_path = DATA_PATH / "dataset4.dat"
pd_std1_path = DATA_PATH / "Pd_std1.dat"

mvsh1_dat = DatFile(mvsh1_path)


@pytest.mark.parametrize(
    "num,digits", [(0, 0), (0.5, 1), (1, 0), (1.25, 2), (2.123, 3), (-1, 0), (-1.2, 1)]
)
def test_num_digits_after_decimal(num, digits):
    assert _num_digits_after_decimal(num) == digits


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


class TestGetAllMvsHInFile:
    def test_uncommented_file(self):
        mvsh2_objs = MvsH.get_all_in_file(mvsh2_path)
        mvsh2_5 = MvsH(mvsh2_path, 5)
        mvsh2_300 = MvsH(mvsh2_path, 300)
        assert len(mvsh2_objs) == 2
        assert mvsh2_objs[0].temperature == 5
        assert mvsh2_objs[1].temperature == 300
        assert (
            mvsh2_objs[0]
            .data["uncorrected_moment"]
            .equals(mvsh2_5.data["uncorrected_moment"])
        )
        assert (
            mvsh2_objs[1]
            .data["uncorrected_moment"]
            .equals(mvsh2_300.data["uncorrected_moment"])
        )

    def test_commented_file(self):
        mvsh2_objs = MvsH.get_all_in_file(mvsh2c_path)
        mvsh2_5 = MvsH(mvsh2c_path, 5)
        mvsh2_300 = MvsH(mvsh2c_path, 300)
        assert len(mvsh2_objs) == 2
        assert mvsh2_objs[0].temperature == 5
        assert mvsh2_objs[1].temperature == 300
        assert (
            mvsh2_objs[0]
            .data["uncorrected_moment"]
            .equals(mvsh2_5.data["uncorrected_moment"])
        )
        assert (
            mvsh2_objs[1]
            .data["uncorrected_moment"]
            .equals(mvsh2_300.data["uncorrected_moment"])
        )
