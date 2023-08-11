import pandas as pd
from magnetopy.experiments.dc_experiment import DcExperiment


def num_digits_after_decimal(number: int | float) -> int:
    """Return the number of digits after the decimal point in a number.

    Parameters
    ----------
    number : int | float

    Returns
    -------
    int
        The number of digits after the decimal point in the number.
    """
    if isinstance(number, int):
        return 0
    return len(str(number).split(".")[1])


def add_uncorrected_moment_columns(experiment: DcExperiment) -> None:
    """`DataFrame`s from .dat files containing dc data have different columns for the
    moment depending on whether the measurement was done in VSM or DC mode. This
    function adds a column called "uncorrected_moment" and "uncorrected_moment_err"
    that contains the moment and moment error from the .dat file, regardless of
    whether the measurement was done in VSM or DC mode.

    Parameters
    ----------
    experiment : DcExperiment
        A `DcExperiment` object (likely either a `MvsH`, `ZFC`, or `FC` object).
    """
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
    """Adds columns to the `data` attribute of a `DcExperiment` object that contain
    the magnetic moment, magnetic susceptibility, and magnetic susceptibility times
    temperature. The columns added are `"moment"`, `"moment_err"`, `"chi"`,
    `"chi_err"`, `"chi_t"`, and `"chi_t_err"`. The units of these values depend on
    the values of the `mass`, `eicosane_mass`, `molecular_weight`, and
    `diamagnetic_correction`. A record of what scaling was applied is added to the
    `scaling` attribute of the `DcExperiment` object.

    Here are the currently supported scaling options:

    - If `mass` is given but not `molecular_weight`, the only available scaling is
    a mass correction.

    - If `mass` and `molecular` weight are given, a molar correction is applied. The
    molar correction can be further modified by giving `eicosane_mass` and/or
    `diamagnetic_correction`.

    Parameters
    ----------
    experiment : DcExperiment
        A `DcExperiment` object with a `data` attribute that has already been
        processed by `add_uncorrected_moment_columns` and thus has a column called
        "uncorrected_moment" and "uncorrected_moment_err".
    mass : float, optional
        mg of sample, by default 0.
    eicosane_mass : float, optional
        mg of eicosane, by default 0.
    molecular_weight : float, optional
        Molecular weight of the material in g/mol, by default 0.
    diamagnetic_correction : float, optional
        Diamagnetic correction of the material in cm^3/mol, by default 0.
    """
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
