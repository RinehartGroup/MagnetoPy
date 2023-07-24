"""
This code tests the class and functions relating to the field correction protocol
described in https://qdusa.com/siteDocs/appNotes/1500-021.pdf

The tests require calibration files found in the Rinehart group's GitHub repo:
https://github.com/RinehartGroup/MagnetoPyCalibration.git, which are installed via
`mp-calibration https://github.com/RinehartGroup/MagnetoPyCalibration.git`, which will
install the calibration files in the user's home directory.
"""

import inspect
from pathlib import Path

from magnetopy.experiments import MvsH, TrueFieldCorrection

TESTS_PATH = Path(inspect.getfile(inspect.currentframe())).parent
DATA_PATH = TESTS_PATH / "data"

pd_std1_path = DATA_PATH / "Pd_std1.dat"
mvsh6_path = DATA_PATH / "mvsh6.dat"


def test_get_data_file():
    """
    The following should all point to the same file, either by direct path, or by
    looking in the .magnetopy directory for the sequence name or file name.
    """
    tfc1 = TrueFieldCorrection(pd_std1_path)
    tfc2 = TrueFieldCorrection("sequence_1")
    tfc3 = TrueFieldCorrection("mvsh_seq1.dat")
    assert tfc1.data.equals(tfc2.data)
    assert tfc1.data.equals(tfc3.data)


def test_get_mass():
    tfc = TrueFieldCorrection("sequence_1")
    assert tfc.pd_mass == 260.4


def test_mvsh_true_field():
    tfc = TrueFieldCorrection("sequence_1")
    mvsh = MvsH(mvsh6_path)
    mvsh.correct_field("sequence_1")
    assert mvsh.data["true_field"].equals(tfc.data["true_field"])


def test_field_correction_file():
    mvsh = MvsH(mvsh6_path)
    mvsh.correct_field("sequence_1")
    assert mvsh.field_correction_file == "mvsh_seq1.dat"
