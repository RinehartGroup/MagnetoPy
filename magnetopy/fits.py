from typing import Tuple

import numpy as np
import pandas as pd
from numpy import arctan, cos, pi, sin
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from magnetopy.calibration import Calibration
from magnetopy.parse_qd import AnalyzedSingleRawDCScan, QDFile, SingleRawDCScan


def determine_blocking_temp(
    zfc: QDFile, fc: QDFile, yval: str = "DC Moment Free Ctr (emu)"
) -> Tuple[float, pd.DataFrame]:
    """
    Returns blocking temperature as determined by
    taking derivative of the difference between the ZFC and FC
    """
    min_temp = max(
        zfc.parsed_data["Temperature (K)"].min(),
        fc.parsed_data["Temperature (K)"].min(),
    )
    max_temp = min(
        zfc.parsed_data["Temperature (K)"].max(),
        fc.parsed_data["Temperature (K)"].max(),
    )
    num_points = min([len(zfc.parsed_data), len(fc.parsed_data)])
    zfc_interp = interp1d(
        zfc.parsed_data["Temperature (K)"],
        zfc.parsed_data[yval],
        kind="cubic",
        fill_value="extrapolate",
    )
    fc_interp = interp1d(
        fc.parsed_data["Temperature (K)"],
        fc.parsed_data[yval],
        kind="cubic",
        fill_value="extrapolate",
    )
    diff = zfc_interp(np.linspace(min_temp, max_temp, num_points)) - fc_interp(
        np.linspace(min_temp, max_temp, num_points)
    )
    df = pd.DataFrame(
        {
            "Temperature (K)": np.linspace(min_temp, max_temp, num_points),
            "FC - ZFC (emu)": diff,
        }
    )
    deriv = np.diff(df["FC - ZFC (emu)"]) / np.diff(df["Temperature (K)"])
    df["deriv"] = np.append(deriv, np.nan)
    blocking_temp = df["Temperature (K)"][df["deriv"].idxmax()]
    return blocking_temp, df


def arctan_fit(
    file: QDFile,
    normalized: bool = False,
    moment_per_mass: bool = False,
) -> Tuple[Tuple[float, float, float, float], pd.DataFrame]:

    mvsh = pd.DataFrame()
    mvsh["field"] = file.parsed_data["Magnetic Field (Oe)"]
    if normalized:
        mvsh["moment"] = (
            file.parsed_data["DC Moment Free Ctr (emu)"]
            / file.parsed_data["DC Moment Free Ctr (emu)"].max()
        )
    elif moment_per_mass:
        mvsh["moment"] = file.parsed_data["Moment_per_mass"]
    else:
        mvsh["moment"] = file.parsed_data["DC Moment Free Ctr (emu)"]

    deriv = np.diff(mvsh["moment"]) / np.diff(mvsh["field"])
    deriv = np.append(deriv, np.nan)
    mvsh["deriv"] = deriv
    mvsh["direction"] = mvsh["field"].diff()

    idx = mvsh[mvsh["direction"] > 0]["field"].idxmax()
    mvsh_up = mvsh[: idx + 1]
    mvsh_down = mvsh[idx:]

    def arc_mvsh(h, chi_p, Ms, hc, w):
        return chi_p * h + (2 * Ms / pi) * arctan((2 * (h - hc)) / w)

    popt_up, pcov_up = curve_fit(
        arc_mvsh,
        mvsh_up["field"],
        mvsh_up["moment"],
        p0=[mvsh["deriv"][0], mvsh["moment"].max(), 0, 1],
        bounds=(
            [-np.inf, mvsh["moment"].max() * 0.8, mvsh["field"].min(), 0],
            [np.inf, mvsh["moment"].max() * 1.2, mvsh["field"].max(), np.inf],
        ),
    )

    popt_down, pcov_down = curve_fit(
        arc_mvsh,
        mvsh_down["field"],
        mvsh_down["moment"],
        p0=[mvsh["deriv"][0], mvsh["moment"].max(), 0, 1],
        bounds=(
            [-np.inf, mvsh["moment"].max() * 0.8, mvsh["field"].min(), 0],
            [np.inf, mvsh["moment"].max() * 1.2, mvsh["field"].max(), np.inf],
        ),
    )

    popt = [
        (popt_up[0] + popt_down[0]) / 2,
        (popt_up[1] + popt_down[1]) / 2,
        (abs(popt_up[2]) + abs(popt_down[2])) / 2,
        (popt_up[3] + popt_down[3]) / 2,
    ]
    popt = tuple(popt)

    h_up = np.linspace(mvsh["field"].min(), mvsh["field"].max(), 500)
    h = np.append(h_up, h_up[-2::-1])
    fit_moment_up = arc_mvsh(h[:500], *popt_up)
    fit_moment_down = arc_mvsh(h[500:], *popt_down)
    fit_moment = np.append(fit_moment_up, fit_moment_down)
    fit_df = pd.DataFrame({"field": h, "moment": fit_moment})

    return popt, fit_df
