from dataclasses import dataclass
import inspect
from pathlib import Path

import pytest

from magnetopy import DatFile, MvsH
from magnetopy.experiments import _num_digits_after_decimal

TESTS_PATH = Path(inspect.getfile(inspect.currentframe())).parent
DATA_PATH = TESTS_PATH / "data"

mvsh1_dat = DatFile(DATA_PATH / "mvsh1.dat")
mvsh2_dat = DatFile(DATA_PATH / "mvsh2.dat")
mvsh2a_dat = DatFile(DATA_PATH / "mvsh2a.dat")
mvsh2b_dat = DatFile(DATA_PATH / "mvsh2b.dat")
mvsh3_dat = DatFile(DATA_PATH / "mvsh3.dat")
mvsh4_dat = DatFile(DATA_PATH / "mvsh4.dat")
mvsh5_dat = DatFile(DATA_PATH / "mvsh5.dat")
mvsh5rw_dat = DatFile(DATA_PATH / "mvsh5.rw.dat")
mvsh6_dat = DatFile(DATA_PATH / "mvsh6.dat")
mvsh7_dat = DatFile(DATA_PATH / "mvsh7.dat")
mvsh8_dat = DatFile(DATA_PATH / "mvsh8.dat")
mvsh9_dat = DatFile(DATA_PATH / "mvsh9.dat")
mvsh10_dat = DatFile(DATA_PATH / "mvsh10.dat")
mvsh11_dat = DatFile(DATA_PATH / "mvsh11.dat")
zfcfc1_dat = DatFile(DATA_PATH / "zfcfc1.dat")
zfcfc2_dat = DatFile(DATA_PATH / "zfcfc2.dat")
zfcfc3_dat = DatFile(DATA_PATH / "zfcfc3.dat")
zfcfc4_dat = DatFile(DATA_PATH / "zfcfc4.dat")
fc4a_dat = DatFile(DATA_PATH / "fc4a.dat")
fc4b_dat = DatFile(DATA_PATH / "fc4b.dat")
zfc4a_dat = DatFile(DATA_PATH / "zfc4a.dat")
zfc4b_dat = DatFile(DATA_PATH / "zfc4b.dat")
dataset4_dat = DatFile(DATA_PATH / "dataset4.dat")
pd_std1_dat = DatFile(DATA_PATH / "Pd_std1.dat")


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
    ((mvsh1_dat, 2), _ExpectedMvsH(2, 70000, 70000)),
    ((mvsh1_dat, 4), _ExpectedMvsH(4, 70000, 70000)),
    ((mvsh1_dat, 6), _ExpectedMvsH(6, 70000, 70000)),
    ((mvsh1_dat, 8), _ExpectedMvsH(8, 70000, 70000)),
    ((mvsh1_dat, 10), _ExpectedMvsH(10, 70000, 70000)),
    ((mvsh1_dat, 12), _ExpectedMvsH(12, 70000, 70000)),
    ((mvsh1_dat, 300), _ExpectedMvsH(300, 70000, -70000)),
    ((mvsh2_dat, 5), _ExpectedMvsH(5, -70000, -70000)),
    ((mvsh2_dat, 300), _ExpectedMvsH(300, -70000, -70000)),
    ((mvsh2a_dat, 5), _ExpectedMvsH(5, -70000, -70000)),
    ((mvsh2b_dat, 300), _ExpectedMvsH(300, -70000, -70000)),
    ((mvsh3_dat, 5), _ExpectedMvsH(5, -70000, -70000)),
    ((mvsh6_dat, 300), _ExpectedMvsH(300, -70000, -70000)),
    ((mvsh6_dat, 300.0), _ExpectedMvsH(300.0, -70000, -70000)),
    ((mvsh7_dat, 300), _ExpectedMvsH(300, -70000, -70000)),
    ((mvsh7_dat, 300.1), _ExpectedMvsH(300.1, -70000, -70000)),
    ((mvsh8_dat, 2), _ExpectedMvsH(2, 0, 70000)),
    ((mvsh8_dat, 2.0), _ExpectedMvsH(2.0, 0, 70000)),
    ((mvsh9_dat, 2), _ExpectedMvsH(2, 0, 70000)),
    ((mvsh10_dat, 5), _ExpectedMvsH(5, -70000, -70000)),
    ((mvsh11_dat, 5), _ExpectedMvsH(5, 0, 70000)),
    ((pd_std1_dat, 300), _ExpectedMvsH(300, -70000, -70000)),
    ((pd_std1_dat, 300.1), _ExpectedMvsH(300.1, -70000, -70000)),
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
    ((mvsh4_dat, 293), _ExpectedMvsH(293, -70000, -70000)),
    ((mvsh5_dat, 293), _ExpectedMvsH(293, -70000, -70000)),
    ((dataset4_dat, 293), _ExpectedMvsH(293, -70000, -70000)),
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
    ((mvsh2a_dat, None), 5),
    ((mvsh2b_dat, None), 300),
    ((mvsh3_dat, None), 5),
    ((mvsh4_dat, None), 293),
    ((dataset4_dat, None), 293),
    ((mvsh5_dat, None), 293),
    ((mvsh6_dat, None), 300),
    ((mvsh7_dat, None), 300),
    ((mvsh7_dat, 1), 300.1),
    ((mvsh8_dat, None), 2),
    ((mvsh9_dat, None), 2),
    ((mvsh10_dat, None), 5),
    ((mvsh11_dat, None), 5),
    ((pd_std1_dat, None), 300),
    ((pd_std1_dat, 1), 300.1),
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
    mvsh1_dat,
    mvsh2_dat,
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
    (MvsH(mvsh1_dat, 2), _ExpectedSegment(None, (70000, -70000), (-70000, 70000))),
    (MvsH(mvsh1_dat, 4), _ExpectedSegment(None, (70000, -70000), (-70000, 70000))),
    (MvsH(mvsh1_dat, 6), _ExpectedSegment(None, (70000, -70000), (-70000, 70000))),
    (MvsH(mvsh1_dat, 8), _ExpectedSegment(None, (70000, -70000), (-70000, 70000))),
    (MvsH(mvsh1_dat, 10), _ExpectedSegment(None, (70000, -70000), (-70000, 70000))),
    (MvsH(mvsh1_dat, 12), _ExpectedSegment(None, (70000, -70000), (-70000, 70000))),
    (MvsH(mvsh1_dat, 300), _ExpectedSegment(None, (70000, -70000), None)),
    (MvsH(mvsh2_dat, 5), _ExpectedSegment(None, (70000, -70000), (-70000, 70000))),
    (MvsH(mvsh2_dat, 300), _ExpectedSegment(None, (70000, -70000), (-70000, 70000))),
    (MvsH(mvsh3_dat, 5), _ExpectedSegment(None, (70000, -70000), (-70000, 70000))),
    (MvsH(mvsh4_dat, 293), _ExpectedSegment(None, (70000, -70000), (-70000, 70000))),
    (MvsH(mvsh5_dat, 293), _ExpectedSegment(None, (70000, -70000), (-70000, 70000))),
    (MvsH(mvsh6_dat, 300), _ExpectedSegment(None, (70000, -70000), (-70000, 70000))),
    (MvsH(mvsh7_dat, 300), _ExpectedSegment(None, (70000, -70000), (-70000, 70000))),
    (
        MvsH(mvsh8_dat, 2),
        _ExpectedSegment((0, 70039), (70039, -70026), (-70026, 70000)),
    ),
    (
        MvsH(mvsh9_dat, 2),
        _ExpectedSegment((0, 70000), (70000, -70000), (-70000, 70000)),
    ),
    (MvsH(mvsh10_dat, 5), _ExpectedSegment(None, (70000, -70000), (-70000, 70000))),
    (
        MvsH(mvsh11_dat, 5),
        _ExpectedSegment((0, 70000), (70000, -70000), (-70000, 70000)),
    ),
    (MvsH(pd_std1_dat, 300), _ExpectedSegment(None, (70000, -70000), (-70000, 70000))),
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
