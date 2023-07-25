import inspect
from pathlib import Path
from dataclasses import dataclass

import pytest

from magnetopy.dataset import Dataset, SampleInfo

TESTS_PATH = Path(inspect.getfile(inspect.currentframe())).parent
DATA_PATH = TESTS_PATH / "data"

mvsh2_path = DATA_PATH / "mvsh2.dat"
mvsh9_path = DATA_PATH / "mvsh9.dat"

### SampleInfo tests ###


@dataclass
class _ExpectedSampleInfo:
    material: str | None
    comment: str | None
    mass: float | None
    volume: float | None
    molecular_weight: float | None
    size: float | None
    shape: str | None
    holder: str | None
    holder_detail: str | None
    offset: float | None
    eicosane_mass: float | None
    diamagnetic_correction: float | None


sample_info_expected = [
    (
        mvsh2_path,
        _ExpectedSampleInfo(
            material="KMK282",
            comment=None,
            mass=0.7,
            volume=None,
            molecular_weight=None,
            size=None,
            shape=None,
            holder="Standard",
            holder_detail="Standard",
            offset=65.99,
            eicosane_mass=None,
            diamagnetic_correction=None,
        ),
    ),
    (
        mvsh9_path,
        _ExpectedSampleInfo(
            material="[Er_TMSCOT_Cl_THF]2",
            comment="pink crystalline powder",
            mass=9.3,
            volume=None,
            molecular_weight=1046.664,
            size=None,
            shape=None,
            holder="Standard",
            holder_detail="Standard",
            offset=66.22,
            eicosane_mass=19.4,
            diamagnetic_correction=-0.00050736,
        ),
    ),
]


@pytest.mark.parametrize("path,expected", sample_info_expected)
def test_sample_info_w_field_hack(path: str, expected: _ExpectedSampleInfo):
    sample_info = SampleInfo.from_dat_file(path)
    for item in expected.__dict__:
        assert getattr(sample_info, item) == getattr(expected, item)


sample_info_expected = [
    (
        mvsh2_path,
        _ExpectedSampleInfo(
            material="KMK282",
            comment=None,
            mass=0.7,
            volume=None,
            molecular_weight=None,
            size=None,
            shape=None,
            holder="Standard",
            holder_detail="Standard",
            offset=65.99,
            eicosane_mass=None,
            diamagnetic_correction=None,
        ),
    ),
    (
        mvsh9_path,
        _ExpectedSampleInfo(
            material="[Er_TMSCOT_Cl_THF]2",
            comment="pink crystalline powder",
            mass=9.3,
            volume=19.4,
            molecular_weight=1046.664,
            size=-0.00050736,
            shape=None,
            holder="Standard",
            holder_detail="Standard",
            offset=66.22,
            eicosane_mass=None,
            diamagnetic_correction=None,
        ),
    ),
]


@pytest.mark.parametrize("path,expected", sample_info_expected)
def test_sample_info_no_field_hack(path: str, expected: _ExpectedSampleInfo):
    sample_info = SampleInfo.from_dat_file(path, False)
    for item in expected.__dict__:
        assert getattr(sample_info, item) == getattr(expected, item)


### Dataset tests ###

expected_experiment_lengths = [
    ("dataset1", (7, 1, 1)),
    ("dataset2", (1, 1, 1)),
    ("dataset3", (1, 2, 2)),
]


@pytest.mark.parametrize("dataset_name,expected", expected_experiment_lengths)
def test_dataset_experiment_lengths(dataset_name: str, expected: tuple[int, int, int]):
    dataset = Dataset(DATA_PATH / dataset_name)
    assert len(dataset.mvsh) == expected[0]
    assert len(dataset.zfc) == expected[1]
    assert len(dataset.fc) == expected[2]


def test_dataset_id_default():
    dataset = Dataset(DATA_PATH / "dataset1")
    assert dataset.sample_id == "dataset1"


def test_dataset_id_custom():
    dataset = Dataset(DATA_PATH / "dataset1", "custom_id")
    assert dataset.sample_id == "custom_id"
