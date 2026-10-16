from pathlib import Path

import numpy as np

from magnetopy import DatFile
from magnetopy.calibration import (
    determine_moment_calibration,
    fit_mpms3_dc_voltage,
    mpms3_dc_voltage,
)


def test_fit_mpms3_dc_voltage_recovers_known_parameters():
    position = np.linspace(14, 49, 200)
    voltage = mpms3_dc_voltage(position, 0.001, -125.0, 31.7)

    fit = fit_mpms3_dc_voltage(position, voltage)

    assert round(fit.offset, 6) == 0.001
    assert round(fit.amplitude, 6) == -125.0
    assert round(fit.center, 6) == 31.7


def test_determine_moment_calibration_from_dat_file(tmp_path: Path):
    dat_path = tmp_path / "moment.dat"
    rw_path = tmp_path / "moment.rw.dat"
    position = np.linspace(14, 49, 21)
    free_voltage = mpms3_dc_voltage(position, 0, -125.0, 31.7)
    fixed_voltage = mpms3_dc_voltage(position, 0, -124.0, 31.7)

    dat_path.write_text(
        "\n".join(
            [
                "[Header]",
                "; MPMS3 Data File (default extension .dat)",
                "TITLE,",
                "FILEOPENTIME,0,01/01/2024,12:00 pm",
                "[Data]",
                (
                    "Comment,Time Stamp (sec),Temperature (K),Magnetic Field (Oe),"
                    "Moment (emu),M. Std. Err. (emu),Range,"
                    "DC Moment Fixed Ctr (emu),DC Moment Err Fixed Ctr (emu),"
                    "DC Moment Free Ctr (emu),DC Moment Err Free Ctr (emu)"
                ),
                ",0,300,50,,,10,-0.000248,0,-0.00025,0",
            ]
        ),
        encoding="utf-8",
    )
    rw_path.write_text(
        _raw_dat_text(position, free_voltage, fixed_voltage),
        encoding="utf-8",
    )

    dat_file = DatFile(dat_path, parse_raw=True)
    calibration = determine_moment_calibration(dat_file)

    assert len(calibration.records) == 1
    assert round(calibration.records[0].amplitude, 6) == -125.0
    assert round(calibration.calibration_factor_by_range[10], 12) == 0.000002
    assert round(calibration.range_normalized_calibration_factor, 13) == 0.0000002
    assert round(calibration.moment_from_amplitude(-125.0, 10), 8) == -0.00025
    assert (
        round(calibration.moment_from_range_scaled_amplitude(-1250.0), 8)
        == -0.00025
    )


def test_determine_moment_calibration_fixed_center(tmp_path: Path):
    dat_path = tmp_path / "moment.dat"
    rw_path = tmp_path / "moment.rw.dat"
    position = np.linspace(14, 49, 21)
    free_voltage = mpms3_dc_voltage(position, 0, -125.0, 31.7)
    fixed_voltage = mpms3_dc_voltage(position, 0, -124.0, 31.7)

    dat_path.write_text(
        "\n".join(
            [
                "[Header]",
                "; MPMS3 Data File (default extension .dat)",
                "TITLE,",
                "FILEOPENTIME,0,01/01/2024,12:00 pm",
                "[Data]",
                (
                    "Comment,Time Stamp (sec),Temperature (K),Magnetic Field (Oe),"
                    "Moment (emu),M. Std. Err. (emu),Range,"
                    "DC Moment Fixed Ctr (emu),DC Moment Err Fixed Ctr (emu),"
                    "DC Moment Free Ctr (emu),DC Moment Err Free Ctr (emu)"
                ),
                ",0,300,50,,,10,-0.000248,0,-0.00025,0",
            ]
        ),
        encoding="utf-8",
    )
    rw_path.write_text(
        _raw_dat_text(position, free_voltage, fixed_voltage),
        encoding="utf-8",
    )

    calibration = determine_moment_calibration(
        DatFile(dat_path, parse_raw=True), "fixed"
    )

    assert round(calibration.records[0].amplitude, 6) == -124.0
    assert round(calibration.calibration_factor_by_range[10], 12) == 0.000002


def _raw_dat_text(
    position: np.ndarray,
    free_voltage: np.ndarray,
    fixed_voltage: np.ndarray,
) -> str:
    header = [
        "[Header]",
        "; DC Raw Data File (default extension .raw)",
        "TITLE,",
        "FILEOPENTIME,0,01/01/2024,12:00 pm",
        "[Data]",
        (
            "Comment,Time Stamp (sec),Raw Position (mm),Raw Voltage (V),"
            "Processed Voltage (V),Fixed C Fitted (V),Free C Fitted (V)"
        ),
        _scan_header(10),
    ]
    up_rows = [
        f",{idx},{x},0,{free},,"
        for idx, (x, free) in enumerate(zip(position, free_voltage), start=1)
    ]
    down_header = [_scan_header(10)]
    down_rows = [
        f",{idx},{x},0,{free},,"
        for idx, (x, free) in enumerate(
            zip(position[::-1], free_voltage[::-1]),
            start=len(up_rows) + 2,
        )
    ]
    fit_rows = [
        f",{idx},{x},,,{fixed},{free}"
        for idx, (x, fixed, free) in enumerate(
            zip(position, fixed_voltage, free_voltage),
            start=len(up_rows) + len(down_rows) + 3,
        )
    ]
    return "\n".join(header + up_rows + down_header + down_rows + fit_rows)


def _scan_header(squid_range: int) -> str:
    return (
        ";low temp = 300.0 K;high temp = 300.0 K;avg. temp = 300.0 K;"
        "low field = 50.0 Oe;high field = 50.0 Oe;drift = 0 V/s;"
        "slope = 0 V/mm;"
        f"squid range = {squid_range};"
        "given center = 31.7 mm;calculated center = 31.7 mm;"
        "amp fixed = -1.0 V;amp free =-1.0 V"
    )
