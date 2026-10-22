"""MPMS3 DC moment calibration from a Pd standard measurement.

Status: not yet validated on real data. The tests use synthetic scans only. A
check against the Rinehart group's Pd standard (2026-10-22) recovered a constant
range-normalized calibration factor across SQUID ranges 1-1000, but that factor is
derived from Quantum Design's own fitted curve and reported moment, so it is
self-consistent by construction. Fitting the measured "Processed Voltage (V)" scans
instead gave amplitudes about 0.2-0.3% larger than fitting Quantum Design's fitted
curve. Moments computed from raw voltages should be compared against reported
moments on real measurements before this is relied on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit

from magnetopy.data_files import DatFile, DcMeasurement


# Values that reproduce Quantum Design's fitted DC scan curves (relative rms residual
# ~1e-7 when fit freely to the Pd standard scans); the nominal 8.5 mm / 8.0 mm left a
# ~0.45% systematic shape mismatch.
MPMS3_GRADIOMETER_RADIUS_MM = 8.3654
MPMS3_GRADIOMETER_HALF_LENGTH_MM = 7.9600

DcCenter = Literal["free", "fixed"]


@dataclass(frozen=True)
class DcFitResult:
    """Fit parameters for an MPMS3 DC voltage scan."""

    offset: float
    amplitude: float
    center: float


@dataclass(frozen=True)
class MomentCalibrationRecord:
    """Moment calibration determined from one Pd standard measurement."""

    field: float
    squid_range: int
    moment: float
    amplitude: float
    calibration_factor: float
    range_normalized_calibration_factor: float
    fit: DcFitResult


@dataclass(frozen=True)
class MomentCalibration:
    """System-specific voltage-to-moment calibration for MPMS3 DC scans."""

    center: DcCenter
    records: tuple[MomentCalibrationRecord, ...]

    @property
    def calibration_factor_by_range(self) -> dict[int, float]:
        """Mean calibration factor for each SQUID range in emu/V."""
        ranges = sorted({record.squid_range for record in self.records})
        return {
            squid_range: float(
                np.mean(
                    [
                        record.calibration_factor
                        for record in self.records
                        if record.squid_range == squid_range
                    ]
                )
            )
            for squid_range in ranges
        }

    @property
    def range_normalized_calibration_factor(self) -> float:
        """Mean range-normalized calibration factor in emu/V."""
        return float(
            np.mean(
                [
                    record.range_normalized_calibration_factor
                    for record in self.records
                ]
            )
        )

    def moment_from_amplitude(self, amplitude: float, squid_range: int) -> float:
        """Convert an unscaled fitted amplitude to magnetic moment in emu."""
        return amplitude * self.calibration_factor_by_range[int(squid_range)]

    def moment_from_range_scaled_amplitude(self, amplitude: float) -> float:
        """Convert a fitted amplitude after multiplying voltage by SQUID range."""
        return amplitude * self.range_normalized_calibration_factor


def mpms3_dc_voltage(
    position: npt.ArrayLike,
    offset: float,
    amplitude: float,
    center: float,
    radius: float = MPMS3_GRADIOMETER_RADIUS_MM,
    half_length: float = MPMS3_GRADIOMETER_HALF_LENGTH_MM,
) -> np.ndarray:
    """Evaluate the MPMS3 second-order gradiometer DC scan voltage model."""
    z = np.asarray(position, dtype=float)
    return offset + amplitude * (
        2 * (radius**2 + (z - center) ** 2) ** -1.5
        - (radius**2 + (half_length + z - center) ** 2) ** -1.5
        - (radius**2 + (-half_length + z - center) ** 2) ** -1.5
    )


def fit_mpms3_dc_voltage(
    position: npt.ArrayLike,
    voltage: npt.ArrayLike,
    center_guess: float | None = None,
) -> DcFitResult:
    """Fit the MPMS3 DC scan voltage model to processed voltage data."""
    z = np.asarray(position, dtype=float)
    y = np.asarray(voltage, dtype=float)
    if z.shape != y.shape:
        raise ValueError("position and voltage must have the same shape.")

    center = float(center_guess if center_guess is not None else np.mean(z))
    offset = float(np.median(y))
    response = mpms3_dc_voltage(z, 0, 1, center)
    response_span = float(np.max(response) - np.min(response))
    voltage_span = float(np.max(y) - np.min(y))
    amplitude = voltage_span / response_span
    if y[np.argmax(response)] < y[np.argmin(response)]:
        amplitude *= -1

    popt, _ = curve_fit(
        mpms3_dc_voltage,
        z,
        y,
        p0=(offset, amplitude, center),
        maxfev=10000,
    )
    return DcFitResult(float(popt[0]), float(popt[1]), float(popt[2]))


def determine_moment_calibration(
    calibration_dat: str | Path | DatFile | None = None,
    center: DcCenter = "free",
) -> MomentCalibration:
    """Determine MPMS3 voltage-to-moment constants from a Pd standard file.

    The calibration factor for each range is the reported DC moment divided by
    the fitted amplitude from the MPMS3 DC voltage model. The range-normalized
    factor is additionally divided by SQUID range, matching the application-note
    workflow where processed voltages are multiplied by range before fitting.

    Parameters
    ----------
    calibration_dat : str | Path | DatFile | None, optional
        The Pd standard ``.dat`` file. If ``None``, the ``"moment"`` file from
        ``~/.magnetopy/calibration/calibration.json`` is used.
    center : {"free", "fixed"}, optional
        Which Quantum Design DC moment and fitted waveform to use.

    Returns
    -------
    MomentCalibration
        The per-measurement records and averaged calibration constants.
    """
    dat_file = _prepare_calibration_dat(calibration_dat)
    moment_column, fit_column = _columns_for_center(center)
    records: list[MomentCalibrationRecord] = []

    for _, row in dat_file.data.dropna(subset=["raw_scan"]).iterrows():
        raw_scan: DcMeasurement = row["raw_scan"]
        fit_scan = raw_scan.fit_scan.data
        center_guess = (
            raw_scan.up.calculated_center
            if center == "free"
            else raw_scan.up.given_center
        )
        fit = fit_mpms3_dc_voltage(
            fit_scan["Raw Position (mm)"],
            fit_scan[fit_column],
            center_guess,
        )
        moment = float(row[moment_column])
        squid_range = int(row["Range"])
        factor = moment / fit.amplitude
        records.append(
            MomentCalibrationRecord(
                field=float(row["Magnetic Field (Oe)"]),
                squid_range=squid_range,
                moment=moment,
                amplitude=fit.amplitude,
                calibration_factor=factor,
                range_normalized_calibration_factor=factor / squid_range,
                fit=fit,
            )
        )

    if not records:
        raise ValueError("No raw DC scans were found in the calibration data.")
    return MomentCalibration(center=center, records=tuple(records))


def _prepare_calibration_dat(calibration_dat: str | Path | DatFile | None) -> DatFile:
    if calibration_dat is None:
        calibration_dat = _installed_moment_calibration_path()

    if isinstance(calibration_dat, DatFile):
        dat_file = calibration_dat
        if "raw_scan" not in dat_file.data.columns:
            rw_dat = dat_file.local_path.with_suffix(".rw.dat")
            if not rw_dat.exists():
                raise FileNotFoundError(f"Could not find raw calibration file: {rw_dat}")
            dat_file.append_raw_data(rw_dat)
        return dat_file

    return DatFile(calibration_dat, parse_raw=True)


def _installed_moment_calibration_path() -> Path:
    calibration_dir = Path.home() / ".magnetopy" / "calibration"
    with (calibration_dir / "calibration.json").open(encoding="utf-8") as f:
        calibration_config = json.load(f)
    return calibration_dir / "calibration_files" / calibration_config["moment"]


def _columns_for_center(center: DcCenter) -> tuple[str, str]:
    if center == "free":
        return "DC Moment Free Ctr (emu)", "Free C Fitted (V)"
    if center == "fixed":
        return "DC Moment Fixed Ctr (emu)", "Fixed C Fitted (V)"
    raise ValueError("center must be either 'free' or 'fixed'.")
