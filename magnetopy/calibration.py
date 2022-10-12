"""
Classes
----------
Calibration
    Determines calibration factors from measurements on Pd standards
"""

import pathlib

import pandas as pd

from magnetopy.parse_qd import QDFile


class Calibration:
    def __init__(self, path: pathlib.Path):
        self.file = QDFile(path)
        self.file.analyze_raw()
        self.factors = self._determine_calibration_factors()

    def __repr__(self):
        return f"Calibration using {self.file}"

    def _determine_calibration_factors(self) -> pd.DataFrame:
        list_dicts = []
        for i, row in self.file.parsed_data.iterrows():
            cal_factor = row["DC Moment Free Ctr (emu)"] / row["Raw Scans"].fit_obj.a
            meas = {
                "Temp (K)": row["Temperature (K)"],
                "Field (Oe)": row["Magnetic Field (Oe)"],
                "Range": row["Range"],
                "Calibration Factor": cal_factor,
            }
            list_dicts.append(meas)
        cal_df = pd.DataFrame(list_dicts)
        return cal_df

    def select_cal_factor(self, range: int, field: float, temp: float) -> float:
        """
        Returns the calibration factor for a set of conditions that most closely match
        the conditions in which the calibration factor was determined,
        prioritizing range, then field, then temp
        """
        cal_df = self.factors.copy()
        cal_df = cal_df[cal_df["Range"] == range]
        cal_df["Field_diff"] = abs(cal_df["Field (Oe)"] - field)
        cal_df["Temp_diff"] = abs(cal_df["Temp (K)"] - temp)
        cal_df = cal_df[cal_df["Field_diff"] < (cal_df["Field_diff"].min() + 1)]
        cal_df = cal_df[cal_df["Temp_diff"] == cal_df["Temp_diff"].min()]
        cal_df = cal_df.reset_index(drop=True)
        cal_factor = cal_df.at[0, "Calibration Factor"]
        return cal_factor
