import numpy as np
import pandas as pd

import pytest

from magnetopy.experiments import (
    _add_uncorrected_moment_columns,
    _scale_dc_data,
    _scale_magnetic_data_mass,
    _scale_magnetic_data_molar_w_eicosane_and_diamagnet,
)


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
