import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from magnetopy import DatFile
from magnetopy.parsing_utils import (
    find_outlier_indices,
    find_sequence_starts,
    find_temp_turnaround_point,
    label_clusters,
    unique_values,
)

TESTS_PATH = Path(inspect.getfile(inspect.currentframe())).parent
DATA_PATH = TESTS_PATH / "data"

# see file_descriptions.md for more info on these files
# the tests here will only look at files that don't have comments and
# therefore need the algorithms found in parsing_utils.py to separate
# the experiments that are found inside
mvsh1 = DatFile(DATA_PATH / "mvsh1.dat")
mvsh2 = DatFile(DATA_PATH / "mvsh2.dat")
mvsh2a = DatFile(DATA_PATH / "mvsh2a.dat")
mvsh2b = DatFile(DATA_PATH / "mvsh2b.dat")
mvsh3 = DatFile(DATA_PATH / "mvsh3.dat")
mvsh6 = DatFile(DATA_PATH / "mvsh6.dat")
mvsh7 = DatFile(DATA_PATH / "mvsh7.dat")
mvsh8 = DatFile(DATA_PATH / "mvsh8.dat")
mvsh9 = DatFile(DATA_PATH / "mvsh9.dat")
mvsh10 = DatFile(DATA_PATH / "mvsh10.dat")
mvsh11 = DatFile(DATA_PATH / "mvsh11.dat")
pd_std1 = DatFile(DATA_PATH / "Pd_std1.dat")
zfcfc1 = DatFile(DATA_PATH / "zfcfc1.dat")
zfcfc2 = DatFile(DATA_PATH / "zfcfc2.dat")
zfcfc3 = DatFile(DATA_PATH / "zfcfc3.dat")


temp_clusters = [
    (mvsh1, np.array([0, 1, 2, 3, 4, 5, 6])),
    (mvsh2, np.array([0, 1])),
    (mvsh2a, np.array([0])),
    (mvsh2b, np.array([0])),
    (mvsh3, np.array([0])),
    (mvsh6, np.array([0])),
    (mvsh7, np.array([0])),
    (mvsh8, np.array([0])),
    (mvsh9, np.array([0])),
    (mvsh10, np.array([0])),
    (mvsh11, np.array([0])),
    (zfcfc1, np.array([0])),
    (pd_std1, np.array([0])),
]


@pytest.mark.parametrize("dat_file,expected", temp_clusters)
def test_label_clusters(dat_file: DatFile, expected: np.ndarray):
    found_clusters = np.unique(label_clusters(dat_file.data["Temperature (K)"]))
    assert np.array_equal(found_clusters, expected)


temperatures_n0 = [
    (mvsh1, [2, 4, 6, 8, 10, 12, 300]),
    (mvsh2, [5, 300]),
    (mvsh2a, [5]),
    (mvsh2b, [300]),
    (mvsh3, [5]),
    (mvsh6, [300]),
    (mvsh7, [300]),
    (mvsh8, [2]),
    (mvsh9, [2]),
    (mvsh10, [5]),
    (mvsh11, [5]),
    (pd_std1, [300]),
]


@pytest.mark.parametrize("dat_file,expected", temperatures_n0)
def test_unique_values(dat_file: DatFile, expected: list[int]):
    found_values = unique_values(dat_file.data["Temperature (K)"])
    assert found_values == expected


temperatures_n1 = [
    (mvsh1, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 300.0]),
    (mvsh2, [5.0, 300.0]),
    (mvsh2a, [5.0]),
    (mvsh2b, [300.0]),
    (mvsh3, [5.0]),
    (mvsh6, [300.0]),
    (mvsh7, [300.1]),
    (mvsh8, [2.0]),
    (mvsh9, [2.0]),
    (mvsh10, [5.0]),
    (mvsh11, [5.0]),
    (pd_std1, [300.1]),
]


@pytest.mark.parametrize("dat_file,expected", temperatures_n1)
def test_unique_values_n1(dat_file: DatFile, expected: list[float]):
    found_values = unique_values(dat_file.data["Temperature (K)"], ndigits=1)
    assert found_values == expected


outliers = [
    (zfcfc1, [252]),
    (zfcfc2, [199]),
    (zfcfc3, []),
]


@pytest.mark.parametrize("dat_file,expected", outliers)
def test_find_outlier_indices(dat_file: DatFile, expected: list[int]):
    """
    While `find_outlier_indices` is a general function, it's currently only used to
    determine the turnaround points in ZFCFC experiments.
    """
    df = dat_file.data.copy()
    df["diff"] = df["Temperature (K)"].diff()
    assert find_outlier_indices(df["diff"]) == expected


turnaround_points = [
    (zfcfc1, 252),
    (zfcfc2, 199),
    (zfcfc3, 256),
]


@pytest.mark.parametrize("dat_file,expected", turnaround_points)
def test_find_temp_turnaround_point(dat_file: DatFile, expected: int):
    assert find_temp_turnaround_point(dat_file.data) == expected


expected_sequences = [
    (([0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4], 0), [0, 6, 11]),
    (([5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1], 0), [0, 6, 11]),
    (([0, 1, 2, 3, 4, 5, 4.5, 5.1, 4, 3, 2, 1], 0), [0, 6, 7]),
    (([0, 1, 2, 3, 4, 5, 4.5, 5.1, 4, 3, 2, 1], 1), [0, 8]),
    (([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], 0), [0, 5]),
    (([4, 3, 2, 1, 0, 4, 3, 2, 1, 0], 0), [0, 5]),
    (
        (
            [
                -70,
                -50,
                -30,
                -10,
                0,
                10,
                30,
                50,
                68,
                70,
                70.1,
                70,
                70.2,
                68,
                50,
                30,
                10,
                0,
                -10,
                -30,
                -50,
                -70,
            ],
            0,
        ),
        [0, 11, 12],
    ),
    (
        (
            [
                -70,
                -50,
                -30,
                -10,
                0,
                10,
                30,
                50,
                68,
                70,
                70.1,
                70,
                70.2,
                68,
                50,
                30,
                10,
                0,
                -10,
                -30,
                -50,
                -70,
            ],
            1,
        ),
        [0, 13],
    ),
    (
        (
            [
                0,
                10,
                20,
                30,
                40,
                50,
                60,
                68,
                70.1,
                69.8,
                68,
                60,
                50,
                40,
                30,
                20,
                10,
                -50,
                -60,
                -69.7,
                -70.2,
                -69.5,
                -60,
                -40,
                -10,
                0.5,
                10,
                30,
                50,
                70,
            ],
            0,
        ),
        [0, 9, 21],
    ),
    (
        (
            [
                0,
                10,
                20,
                30,
                40,
                50,
                69,
                68.5,
                70.1,
                69.8,
                68,
                60,
                50,
                40,
                30,
                20,
                10,
                -50,
                -69.9,
                -69.7,
                -70.2,
                -69.5,
                -60,
                -40,
                -10,
                0.5,
                10,
                30,
                50,
                70,
            ],
            1,
        ),
        [0, 9, 21],
    ),
]


@pytest.mark.parametrize("args,expected", expected_sequences)
def test_find_sequence_starts(args: tuple[list[float], int], expected: list[int]):
    """
    This function is currently only used to read data from an M vs H experiment and
    find the indices of the start of the virgin (if present), reverse, and forward
    field scans.
    """
    ser = pd.Series(args[0])
    assert find_sequence_starts(ser, args[1]) == expected
