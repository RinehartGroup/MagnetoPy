import pandas as pd
from magnetopy.experiments.dc_experiment import DcExperiment


def num_digits_after_decimal(number: int | float):
    if isinstance(number, int):
        return 0
    return len(str(number).split(".")[1])


def add_uncorrected_moment_columns(experiment: DcExperiment) -> None:
    # set "uncorrected_moment" to be the moment directly from the dat file
    # whether the measurement was dc or vsm
    experiment.data["uncorrected_moment"] = experiment.data["Moment (emu)"].fillna(
        experiment.data["DC Moment Free Ctr (emu)"]
    )
    experiment.data["uncorrected_moment_err"] = experiment.data[
        "M. Std. Err. (emu)"
    ].fillna(experiment.data["DC Moment Err Free Ctr (emu)"])


def scale_dc_data(
    experiment: DcExperiment,
    mass: float = 0,
    eicosane_mass: float = 0,
    molecular_weight: float = 0,
    diamagnetic_correction: float = 0,
) -> None:
    mass = mass / 1000  # convert to g
    if mass and molecular_weight:
        experiment.scaling.append("molar")
        if eicosane_mass:
            experiment.scaling.append("eicosane")
        if diamagnetic_correction:
            experiment.scaling.append("diamagnetic_correction")
        mol = mass / molecular_weight
        _scale_magnetic_data_molar_w_eicosane_and_diamagnet(
            experiment.data, mol, eicosane_mass, diamagnetic_correction
        )
    elif mass:
        experiment.scaling.append("mass")
        _scale_magnetic_data_mass(experiment.data, mass)


def _scale_magnetic_data_molar_w_eicosane_and_diamagnet(
    data: pd.DataFrame,
    mol_sample: float,
    eicosane_mass: float,
    diamagnetic_correction: float,
) -> None:
    mol_eicosane = (eicosane_mass / 1000) / 282.55 if eicosane_mass else 0
    eicosane_diamagnetism = (
        -0.00024306 * mol_eicosane
    )  # eicosane chi_D = -0.00024306 emu/mol
    sample_molar_diamagnetism = (
        mol_sample * diamagnetic_correction if diamagnetic_correction else 0
    )
    # chi in units of cm^3/mol
    data["chi"] = (
        data["uncorrected_moment"] / data["Magnetic Field (Oe)"] - eicosane_diamagnetism
    ) / mol_sample - sample_molar_diamagnetism
    data["chi_err"] = abs(
        (
            data["uncorrected_moment_err"] / data["Magnetic Field (Oe)"]
            - eicosane_diamagnetism
        )
        / mol_sample
        - sample_molar_diamagnetism
    )
    # chiT in units of cm3 K mol-1
    data["chi_t"] = data["chi"] * data["Temperature (K)"]
    data["chi_t_err"] = data["chi_err"] * data["Temperature (K)"]
    # moment in units of Bohr magnetons
    data["moment"] = data["chi"] * data["Magnetic Field (Oe)"] / 5585
    data["moment_err"] = abs(data["chi_err"] * data["Magnetic Field (Oe)"] / 5585)


def _scale_magnetic_data_mass(data: pd.DataFrame, mass: float) -> None:
    # moment in units of emu/g
    data["moment"] = data["uncorrected_moment"] / mass
    data["moment_err"] = data["uncorrected_moment_err"] / mass
    # chi in units of cm^3/g
    data["chi"] = data["moment"] / data["Magnetic Field (Oe)"]
    data["chi_err"] = abs(data["moment_err"] / data["Magnetic Field (Oe)"])
    # chiT in units of cm3 K g-1
    data["chi_t"] = data["chi"] * data["Temperature (K)"]
    data["chi_t_err"] = data["chi_err"] * data["Temperature (K)"]
