import inspect
from pathlib import Path
from dataclasses import dataclass

import pytest

from magnetopy.dataset import Dataset, SampleInfo
from magnetopy.experiments import FC, ZFC, MvsH

TESTS_PATH = Path(inspect.getfile(inspect.currentframe())).parent
DATA_PATH = TESTS_PATH / "data"

mvsh2_path = DATA_PATH / "mvsh2.dat"
mvsh9_path = DATA_PATH / "mvsh9.dat"

datasets = ["dataset1", "dataset2", "dataset3", "dataset4"]

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


### Dataset creation tests ###

expected_experiment_lengths = [
    ("dataset1", (7, 1, 1)),
    ("dataset2", (1, 1, 1)),
    ("dataset3", (1, 2, 2)),
    ("dataset4", (1, 1, 1)),
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


### Dataset Scaling Tests ###

expected_auto_scaling = [
    ("dataset1", ["molar", "eicosane"]),
    ("dataset2", ["mass"]),
    ("dataset3", ["mass"]),
    ("dataset4", ["molar", "eicosane", "diamagnetic_correction"]),
]


@pytest.mark.parametrize("dataset_name,expected", expected_auto_scaling)
def test_auto_scaling(dataset_name: str, expected: list[str]):
    dataset = Dataset(DATA_PATH / dataset_name)
    assert dataset.mvsh[0].scaling == expected


@pytest.mark.parametrize("dataset_name", datasets)
def test_mass_scaling(dataset_name: str):
    dataset = Dataset(DATA_PATH / dataset_name, magnetic_data_scaling="mass")
    assert dataset.mvsh[0].scaling == ["mass"]


def test_molar_scaling():
    dataset = Dataset(DATA_PATH / "dataset4", magnetic_data_scaling="molar")
    assert dataset.mvsh[0].scaling == ["molar"]


def test_eicosane_scaling():
    dataset = Dataset(
        DATA_PATH / "dataset4", magnetic_data_scaling=["molar", "eicosane"]
    )
    assert dataset.mvsh[0].scaling == ["molar", "eicosane"]


def test_diamagnetic_correction_scaling():
    dataset = Dataset(
        DATA_PATH / "dataset4",
        magnetic_data_scaling=["molar", "diamagnetic_correction"],
    )
    assert dataset.mvsh[0].scaling == ["molar", "diamagnetic_correction"]


### Dataset Field Correction Tests ###


def test_true_field_correction():
    dataset = Dataset(DATA_PATH / "dataset3", true_field_correction="sequence_1")
    assert dataset.mvsh[0].field_correction_file == "mvsh_seq1.dat"


def test_field_correction_error():
    with pytest.raises(MvsH.FieldCorrectionError):
        Dataset(DATA_PATH / "dataset1", true_field_correction="sequence_1")


### Test Get Experiments ###


def test_get_mvsh():
    dataset = Dataset(DATA_PATH / "dataset1")
    assert (
        dataset.get_mvsh(2)
        .data["uncorrected_moment"]
        .equals(MvsH(DATA_PATH / "dataset1/mvsh1.dat", 2).data["uncorrected_moment"])
    )


def test_get_mvsh_error():
    dataset = Dataset(DATA_PATH / "dataset1")
    with pytest.raises(Dataset.ExperimentNotFoundError):
        dataset.get_mvsh(500)


def test_get_zfc():
    dataset = Dataset(DATA_PATH / "dataset3")
    assert (
        dataset.get_zfc(100)
        .data["uncorrected_moment"]
        .equals(ZFC(DATA_PATH / "dataset3/zfcfc4.dat", 100).data["uncorrected_moment"])
    )


def test_get_zfc_error():
    dataset = Dataset(DATA_PATH / "dataset3")
    with pytest.raises(Dataset.ExperimentNotFoundError):
        dataset.get_zfc(10000)


def test_get_fc():
    dataset = Dataset(DATA_PATH / "dataset3")
    assert (
        dataset.get_fc(1000)
        .data["uncorrected_moment"]
        .equals(FC(DATA_PATH / "dataset3/zfcfc4.dat", 1000).data["uncorrected_moment"])
    )


def test_get_fc_error():
    dataset = Dataset(DATA_PATH / "dataset3")
    with pytest.raises(Dataset.ExperimentNotFoundError):
        dataset.get_fc(10000)
