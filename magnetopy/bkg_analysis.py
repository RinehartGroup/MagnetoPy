import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from magnetopy.calibration import Calibration
from magnetopy.parse_qd import AnalyzedSingleRawDCScan, QDFile, SingleRawDCScan


class QDFitAnalysis:
    """
    Analysis of residuals in fit done by QD software
    """

    def __init__(self):
        pass


class BackgroundSubtraction:
    """
    Subtract Background QDFile from Sample QDFile
    QDFiles must come from identical measurement methods
    """

    def __init__(self, sample: QDFile, background: QDFile, calibration: Calibration):
        pass


def file_background_subtraction(
    sample: QDFile, background: QDFile, calibration: Calibration
):
    data = sample.parsed_data
    data["Background Raw Scans"] = background.parsed_data["Raw Scans"]
    bkg_sub_moment = []
    for meas, bkg in zip(data["Raw Scans", data["Background Raw Scans"]]):
        bkg_sub_moment.append(meas_background_subtraction(meas, bkg, calibration))
    data["Background Subtracted Moment (emu)"] = bkg_sub_moment


def meas_background_subtraction(
    sample: SingleRawDCScan, bkg: SingleRawDCScan, calibration: Calibration
) -> float:
    """
    Takes voltage scans from the sample and background measurements along with a calibration obj and returns the
    background subtracted magnetic moment
    """
    pos = "Raw Position (mm)"
    v = "Processed Voltage (V)"
    start = sample.up_scan[pos].min()
    sample.background_subtracted = pd.DataFrame(
        {"Interp z": np.arange(start, start + 0.175 * 200, 0.175)}
    )

    smp_up = interp1d(
        sample.up_scan[pos], sample.up_scan[v], kind="cubic", fill_value="extrapolate"
    )
    smp_down = interp1d(
        sample.down_scan[pos],
        sample.down_scan[v],
        kind="cubic",
        fill_value="extrapolate",
    )
    bkg_up = interp1d(
        bkg.up_scan[pos], bkg.up_scan[v], kind="cubic", fill_value="extrapolate"
    )
    bkg_down = interp1d(
        bkg.down_scan[pos],
        bkg.down_scan[v],
        kind="cubic",
        fill_value="extrapolate",
    )
    sample.background_subtracted["Interp V"] = (
        smp_up(
            sample.background_subtracted["Interp z"]
            + smp_down.background_subtracted["Interp z"]
        )
        / 2
    ) - (
        bkg_up(
            sample.background_subtracted["Interp z"]
            + bkg_down.background_subtracted["Interp z"]
        )
        / 2
    )
    analyzed_obj = AnalyzedSingleRawDCScan(sample, background_subtracted=True)
    cal_factor = calibration.select_calibration_factor(
        sample.up.squid_range, sample.up.field.low, sample.up.temp.avg
    )
    moment = analyzed_obj.a * cal_factor
    return moment
