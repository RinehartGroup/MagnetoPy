"""
TO DO:
- in QDFile.__init__: add a method to detect the type of file (dc, dc_vsm, ac, etc.) and set attribute QDFile.type
- add a QDFile.mass property that sets the property and adds a "Moment per mass" column to the QDFile.parsed_data based on QDFile.type
"""

"""
Classes
----------
QDFile
    Contains all of the information in a Quantum Design .dat file
    If the directory includes both file.dat and file.rw.dat, the .rd.dat file will also be imported
    as SingleRawDCScan objects and associated with the correct info from the .dat file.
    
    QDFile.parsed_data is the main attribute
SingleRawDCScan
    Contains all of the information for a single scan (up, down, processed) in a .rd.dat file.

    The most important attributes are:
        SingleRawDCScan.proceesed_scan contains the processed scan information
        SingleRawDCScan.info summarizes the parameters (e.g. field, temperature) for the scan
AnalyzedSingleRawDCScan
    Performs a fit of SingleRawDCScan objects to QD equation for a dipole
    moving through a second-order gradiometer
    Modifies the SingleRawDCScan object that was passed to it by adding 
    'Fit' and 'Residual' columns to SingleRawDCScan.processed_scan
"""

import csv
import pathlib
import re
from collections import namedtuple
from typing import NamedTuple

import numpy as np
import pandas as pd
import scipy.integrate as integrate
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


class QDFile:
    """
    Attributes
    ----------
    path: pathlib.Path
        pathlib.Path object of the file (including file name and extension)
    header: list
        All text between '[Header]' and '[Data]' lines
    data: pd.DataFrame
        All data below '[Data]' line
    parsed_data: pd.DataFrame
        Parsed Data organized in dataframe
        If the directory includes both file.dat and file.rw.dat, the .rd.dat file will also be imported
        as SingleRawDCScan objects and associated with the correct entry from the .dat file.


    Methods
    ---------
    analyze_raw:
        If the .dat file is accompanied by a .rw.dat file
        Performs fitting analysis of SingleRawDCScan objects found
        in self.parsed_data
    """

    def __init__(self, path: pathlib.Path):
        self.path = path
        self.header = self._import_header(path)
        self.sample_info = self._sample_info()
        self.data = self._import_data(path, len(self.header) + 2)
        self.comments = self.data["Comment"].dropna().to_list()
        self.data = self.data[self.data["Comment"].isna()].reset_index(drop=True)
        self.raw_header = None
        self.raw_data = None
        if path.with_suffix(".rw.dat").exists():
            self.raw_header = self._import_header(path.with_suffix(".rw.dat"))
            self.raw_data = self._import_data(
                path.with_suffix(".rw.dat"), len(self.raw_header) + 2
            )
        self.parsed_data = self._parse_data()
        self._detect_type()
        self._normalize_moment()

    def __repr__(self):
        return f'File: "{self.path.name}"'

    @staticmethod
    def _import_header(file: pathlib.Path) -> list:
        header = []
        with file.open() as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                if "[Data]" in row:
                    break
                if "[Header]" not in row:
                    header.append(row)
        return header

    def _sample_info(self) -> NamedTuple:
        Info = namedtuple(
            "Info", ["comment", "mass", "volume", "molecular_weight", "size", "shape"]
        )
        for line in self.header:
            if "SAMPLE_COMMENT" in line:
                comment = line[1]
            elif "SAMPLE_MASS" in line:
                if line[1] == "":
                    mass = 0
                else:
                    mass = float(line[1])
            elif "SAMPLE_VOLUME" in line:
                if line[1] == "":
                    volume = 0
                else:
                    volume = float(line[1])
            elif "SAMPLE_MOLECULAR_WEIGHT" in line:
                if line[1] == "":
                    molecular_weight = 0
                else:
                    molecular_weight = float(line[1])
            elif "SAMPLE_SIZE" in line:
                if line[1] == "":
                    size = 0
                else:
                    size = float(line[1])
            elif "SAMPLE_SHAPE" in line:
                shape = line[1]
        return Info(comment, mass, volume, molecular_weight, size, shape)

    @staticmethod
    def _import_data(path: pathlib.Path, skip_rows: int) -> pd.DataFrame:
        return pd.read_csv(str(path), skiprows=skip_rows)

    def _parse_data(self):
        parsed_data = self.data.copy().drop(
            columns=[
                "Transport Action",
                "Pressure (Torr)",
                "Measurement Number",
                "SQUID Status (code)",
                "Motor Status (code)",
                "Measure Status (code)",
                "Motor Current (amps)",
                "Motor Temp. (C)",
                "Temp. Status (code)",
                "Field Status (code)",
                "Chamber Status (code)",
                "Chamber Temp (K)",
                "Redirection State",
                "Evercool Status",
                "DC Scan Length (mm)",
                "DC Scan Time (s)",
                "DC Number of Points",
                "DC Min V (V)",
                "DC Max V (V)",
                "DC Scans per Measure",
            ]
        )
        parsed_data = parsed_data.dropna(axis=1, how="all")
        if isinstance(self.raw_data, pd.DataFrame):
            num_records = int(self.raw_header[-1][3])
            num_magnetic_measurements = int(self.raw_data.shape[0] / num_records)
            parsed_raw = []
            for i in range(num_magnetic_measurements):
                row_num = int(i * num_records)
                data_slice = self.raw_data[row_num : row_num + num_records]
                parsed_raw.append(SingleRawDCScan(data_slice))
            parsed_data["Raw Scans"] = parsed_raw
        return parsed_data

    def analyze_raw(self, background_subtracted: bool = False):
        if not isinstance(self.raw_data, pd.DataFrame):
            return "There are no .rw.dat files to analyze"
        for scan_obj in self.parsed_data["Raw Scans"]:
            scan_obj.analyze(background_subtracted)

    def _detect_type(self):
        if "DC Moment Free Ctr (emu)" in self.parsed_data.columns:
            self.type = "dc_dc"
        elif "Moment (emu)" in self.parsed_data.columns:
            self.type = "dc_vsm"
        else:
            self.type = "unknown or not yet implemented"

    def _normalize_moment(self):
        if self.sample_info.mass != 0:
            if self.type == "dc_vsm":
                self.parsed_data["Moment_per_mass"] = (
                    1000 * self.parsed_data["Moment (emu)"] / self.sample_info.mass
                )
            elif self.type == "dc_dc":
                self.parsed_data["Moment_per_mass"] = (
                    1000
                    * self.parsed_data["DC Moment Free Ctr (emu)"]
                    / self.sample_info.mass
                )


def background_subtraction(sample: QDFile, bkg: QDFile) -> pd.DataFrame:
    for sample_scan_obj, bkg_scan_obj in zip(
        sample.parsed_data["Raw Scans"], bkg.parsed_data["Raw Scans"]
    ):
        # interpolate and average both scans to cover only their shared window
        z = "Raw Position (mm)"
        v = "Processed Voltage (V)"
        min_z, max_z = 100, 0
        for scan in [
            sample_scan_obj.up_scan,
            sample_scan_obj.down_scan,
            bkg_scan_obj.up_scan,
            bkg_scan_obj.down_scan,
        ]:
            min_z = scan[z].min() if scan[z].min() < min_z else min_z
            max_z = scan[z].max() if scan[z].max() > max_z else max_z
        interp_z = np.linspace(min_z, max_z, 200)
        f_up = interp1d(
            sample_scan_obj.up_scan[z],
            sample_scan_obj.up_scan[v],
            kind="cubic",
            fill_value="extrapolate",
        )
        f_down = interp1d(
            sample_scan_obj.down_scan[z],
            sample_scan_obj.down_scan[v],
            kind="cubic",
            fill_value="extrapolate",
        )
        meas_v = (f_up(interp_z) + f_down(interp_z)) / 2
        f_up = interp1d(
            bkg_scan_obj.up_scan[z],
            bkg_scan_obj.up_scan[v],
            kind="cubic",
            fill_value="extrapolate",
        )
        f_down = interp1d(
            bkg_scan_obj.down_scan[z],
            bkg_scan_obj.down_scan[v],
            kind="cubic",
            fill_value="extrapolate",
        )
        bkg_v = (f_up(interp_z) + f_down(interp_z)) / 2
        sample_scan_obj.background_subtracted = pd.DataFrame(
            {
                "Position": interp_z,
                "Measured": meas_v,
                "Background": bkg_v,
            }
        )
        sample_scan_obj.background_subtracted["Subtracted"] = (
            sample_scan_obj.background_subtracted["Measured"]
            - sample_scan_obj.background_subtracted["Background"]
        )
    #     sample_scan.background_subtracted #columns ['position', 'measured', 'background', 'subtracted']
    # sample.parsed_data['Background Subtracted Moment'] = pd.DataFrame


class SingleRawDCScan:
    """
    Each instance holds information from a single DC measurement.
    Instantiation requires a slice of a DataFrame containing a single scan.

    Attributes
    ----------
    up_scan: pd.DataFrame
        Columns: Time Stamp (sec),Raw Position (mm), Raw Voltage (V), Processed Voltage (V)
    down_scan: pd.DataFrame
        Columns: Time Stamp (sec),Raw Position (mm), Raw Voltage (V), Processed Voltage (V)
    fit_scan: pd.DataFrame
        Columns: Time Stamp (sec),Raw Position (mm), Fixed C Fitted (V), Free C Fitted (V)
    processed_scan: pd.DataFrame
        Columns:
            Up z, Up Raw V, Up Drift Corrected V, Up
            Down z, Down Raw V, Down Drift Corrected V,
            Interp z, Up Interp V, Down Interp V, Avg Interp V
    time: float
        Time in seconds of the first entry in the scan using QD timestamp format
    avg_temp: float
        Average temp during entire scan
    avg_field: float
        Average field during entire scan
    up: namedtuple
        Scan values stored DirectionalScan namedtuple. Allows for access
        via self.up.[DirectionalScan_Attribute]
            DirectionalScan Attributes:
                temp: namedtuple
                    temp attributes:
                        high: float
                        low: float
                        avg: float
                            high, low, and avg temp values in K
                field: namedtuple
                    field attributes:
                        high: float
                        low: float
                            high and low values in Oe
                drift: float
                    drift in V/s between down-->up and up-->down scans
                slope: float
                    linear slope in V/mm between down-->up and up-->down scans
                squid_range: int
                given_center: float
                    given in mm
                calculated_center: float
                    given in mm
                amp_fixed: float
                    given in V
                amp_free: float
                    given in V
        Example:
            For a given measurement (up & down scan), one can access the up scan high temp using:
            IndividualRawDCScan.up.temp.high
    down: namedtuple
        same as for self.up


    info: pd.DataFrame
        Formatted summary of scan info
        Actually a @property

    Methods
    ----------
    analyze:
        Carries out fitting of raw scan by creates self.fit_obj,
        which is an AnalyzedRawDCScan instance
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data
        num_records = len(data)
        self.scan_num = int(data.index[0] / num_records)
        self.QD_fit_scan = (
            data[data["Fixed C Fitted (V)"].notna()]
            .copy()
            .drop(columns=["Comment", "Raw Voltage (V)", "Processed Voltage (V)"])
            .reset_index(drop=True)
        )
        data = data[data["Fixed C Fitted (V)"].isna()].drop(
            columns=["Fixed C Fitted (V)", "Free C Fitted (V)"]
        )
        info = data[data["Comment"].notna()]
        info_index = info.index
        self.up_scan = (
            data.loc[info_index[0] + 1 : info_index[1] - 1]
            .copy()
            .drop(columns=["Comment"])
            .reset_index(drop=True)
        )
        self.down_scan = (
            data.loc[info_index[1] + 1 :]
            .copy()
            .drop(columns=["Comment"])
            .reset_index(drop=True)
        )

        self.time = data.iloc[1]["Time Stamp (sec)"]

        # extract info from pre-scan headers
        # attributes defined in _extract_scan_info and
        # a summary of them can be retrieved using self.info
        self._extract_scan_info(info.reset_index(drop=True))

        # process raw scans by accounting for drift and interpolating data
        # into a single scan
        self.processed_scan = self._process_data()

    def __repr__(self):
        return f"Scan {self.scan_num}"

    def __str__(self):
        return f"Scan {self.scan_num}"

    def _extract_scan_info(self, data: pd.DataFrame):
        """
        Processes the info in the pre-scan headers which can then be
        accessed through self.up.PROPERTY and self.down.PROPERTY
        """
        DirectionalScan = namedtuple(
            "DirectionalScan",
            [
                "temp",
                "field",
                "drift",
                "slope",
                "squid_range",
                "given_center",
                "calculated_center",
                "amp_fixed",
                "amp_free",
            ],
        )
        Temp = namedtuple("Temp", ["low", "high", "avg"])
        Field = namedtuple("Field", ["low", "high"])
        str_up_scan_values = data.at[0, "Comment"].split(";")[1:]
        str_down_scan_values = data.at[1, "Comment"].split(";")[1:]
        up_scan_values = []
        down_scan_values = []
        for up, down in zip(str_up_scan_values, str_down_scan_values):
            num_regex = re.compile("-?[0-9]+.?[0-9]*(E)?-?\+?[0-9]*")
            up_scan_values.append(float(num_regex.search(up).group(0)))
            down_scan_values.append(float(num_regex.search(down).group(0)))

        self.up = DirectionalScan(
            Temp(*up_scan_values[0:3]),  # temp
            Field(*up_scan_values[3:5]),  # field
            up_scan_values[5],  # drift
            up_scan_values[6],  # slope
            up_scan_values[7],  # squid_range
            up_scan_values[8],  # given_center
            up_scan_values[9],  # calculated_center
            up_scan_values[10],  # amp_fixed
            up_scan_values[11],  # amp_free
        )

        self.down = DirectionalScan(
            Temp(*down_scan_values[0:3]),  # temp
            Field(*down_scan_values[3:5]),  # field
            down_scan_values[5],  # drift
            down_scan_values[6],  # slope
            down_scan_values[7],  # squid_range
            down_scan_values[8],  # given_center
            down_scan_values[9],  # calculated_center
            down_scan_values[10],  # amp_fixed
            down_scan_values[11],  # amp_free
        )

        self.avg_temp = (self.up.temp.avg + self.down.temp.avg) / 2
        self.avg_field = (
            self.up.field.high
            + self.up.field.low
            + self.down.field.high
            + self.down.field.low
        ) / 4

    def _process_data(self) -> pd.DataFrame:
        """
        Apply drift correction and interpolate to combine up and down scans
        """
        drift = (
            self.down_scan["Raw Voltage (V)"].iloc[-1]
            - self.down_scan["Raw Voltage (V)"].iloc[0]
        ) / (self.down_scan["Time Stamp (sec)"].max() - self.time)
        drift_correction = pd.DataFrame()
        drift_correction["Up z"] = self.up_scan["Raw Position (mm)"]
        drift_correction["Up Raw V"] = self.up_scan["Raw Voltage (V)"]
        drift_correction["Down z"] = self.down_scan["Raw Position (mm)"]
        drift_correction["Down Raw V"] = self.down_scan["Raw Voltage (V)"]
        drift_correction["Up Drift Corrected V"] = self.up_scan["Raw Voltage (V)"] - (
            drift * (self.up_scan["Time Stamp (sec)"] - self.time)
        )
        drift_correction["Down Drift Corrected V"] = self.down_scan[
            "Raw Voltage (V)"
        ] - (drift * (self.down_scan["Time Stamp (sec)"] - self.time))
        f_up = interp1d(
            drift_correction["Up z"],
            drift_correction["Up Drift Corrected V"],
            kind="cubic",
            fill_value="extrapolate",
        )
        f_down = interp1d(
            drift_correction["Down z"],
            drift_correction["Down Drift Corrected V"],
            kind="cubic",
            fill_value="extrapolate",
        )
        processed_scan = pd.DataFrame()
        processed_scan["Interp z"] = np.arange(
            drift_correction["Up z"].min(),
            drift_correction["Up z"].min() + 0.175 * 200,
            0.175,
        )
        processed_scan["Interp V"] = (
            f_up(processed_scan["Interp z"]) + f_down(processed_scan["Interp z"])
        ) / 2
        return processed_scan

    @property
    def info(self):
        """
        Sumarize the info in headers before the up and down scans
        """
        info = pd.DataFrame(
            {
                "Up": [
                    self.up.temp.avg,
                    self.up.temp.low,
                    self.up.temp.high,
                    self.up.field.low,
                    self.up.field.high,
                    self.up.drift,
                    self.up.slope,
                    self.up.squid_range,
                    self.up.given_center,
                    self.up.calculated_center,
                    self.up.amp_fixed,
                    self.up.amp_free,
                ],
                "Down": [
                    self.down.temp.avg,
                    self.down.temp.low,
                    self.down.temp.high,
                    self.down.field.low,
                    self.down.field.high,
                    self.down.drift,
                    self.down.slope,
                    self.down.squid_range,
                    self.down.given_center,
                    self.down.calculated_center,
                    self.down.amp_fixed,
                    self.down.amp_free,
                ],
            },
            index=[
                "Avg Temp (K)",
                "Low Temp (K)",
                "High Temp (K)",
                "Low Field (Oe)",
                "High Field (Oe)",
                "Drift (V/s)",
                "Slope (V/mm)",
                "SQUID Range",
                "Given Center (mm)",
                "Calc Center (mm)",
                "Amp Fixed (V)",
                "Amp Free (V)",
            ],
        )
        return info

    def analyze(self, background_subtracted: bool = False):
        self.fit_obj = AnalyzedSingleRawDCScan(
            self, background_subtracted=background_subtracted
        )

    def analyze_qd_residual(self):
        f_up = interp1d(
            self.up_scan["Raw Position (mm)"],
            self.up_scan["Processed Voltage (V)"],
            kind="cubic",
            fill_value="extrapolate",
        )
        f_down = interp1d(
            self.down_scan["Raw Position (mm)"],
            self.down_scan["Processed Voltage (V)"],
            kind="cubic",
            fill_value="extrapolate",
        )
        qd_res = pd.DataFrame()
        qd_res["Position"] = self.QD_fit_scan["Raw Position (mm)"]
        qd_res["Fit"] = self.QD_fit_scan["Free C Fitted (V)"]
        qd_res["Voltage"] = (f_up(qd_res["Position"]) + f_down(qd_res["Position"])) / 2
        qd_res["Residual"] = qd_res["Voltage"] - qd_res["Fit"]
        return qd_res


class AnalyzedSingleRawDCScan:
    """
    Attributes
    ----------
    scan_obj: SingleRawDCScan
        The SingleRawDCScan instance under analysis
    s, s_err, c, c_err, a, a_err: float
        optimized values and errors from fitting voltage curve
    info: pd.DataFrame
        Contains optimized values and their errors
        as well as residual and information on symmetry of residual
        Actually a @property
    residual_analysis_by_sum: pd.DataFrame
        Integrates residual and gives report on its symmetry (around center, c)
        Actually a @property
    residual_symmetry -> pd.DataFrame:
        The differences in the residual as a function of distance from center, c
        Actually a @property
    Methods
    ----------
    _voltage_curve(self, z: pd.DataFrame, s: float, c: float, a: float)
        Equation reproducing voltage curve in magnetometer
    """

    def __init__(self, scan_obj: SingleRawDCScan, background_subtracted: bool = False):
        self.scan_obj = scan_obj
        scan = (
            scan_obj.processed_scan
            if not background_subtracted
            else scan_obj.background_subtracted
        )
        popt, pcov = curve_fit(
            self._voltage_curve,
            scan["Interp z"],
            scan["Interp V"],
            p0=[scan.at[0, "Interp V"], scan_obj.up.given_center, 0],
        )
        self.s, self.c, self.a = popt
        self.s_err, self.c_err, self.a_err = np.sqrt(np.diag(pcov))
        scan["Fit V"] = self._voltage_curve(scan["Interp z"], self.s, self.c, self.a)
        scan["Residual"] = scan["Interp V"] - scan["Fit V"]

    def __repr__(self):
        return f"Analyzed Scan {self.scan_obj.scan_num}"

    def _voltage_curve(
        self,
        z: pd.DataFrame = pd.DataFrame({"Raw Position (mm)": np.linspace(15, 50, 200)}),
        s: float = None,
        c: float = 34,
        a: float = None,
    ):
        """
        Equation 1 in https://www.qdusa.com/siteDocs/appNotes/1500-023.pdf
        z: sample position in mm
        s: offset voltage (should be "very small")
        c: sample center position
        a: amplitude
        r: radius of gradiometer (8.5 mm)
        l: length of gradiometer (8 mm)
        """
        r = 8.5  # radius of gradiometer in mm
        l = 8  # half the length of the gradiometer in mm
        return s + a * (
            (2 * (r**2 + (z - c) ** 2) ** (-1.5))
            - ((r**2 + (l + z - c) ** 2) ** (-1.5))
            - ((r**2 + (-l + z - c) ** 2) ** (-1.5))
        )

    @property
    def info(self) -> pd.DataFrame:
        """
        Contains optimized values and their errors
        as well as residual and information on symmetry of residual
        """
        scan = self.scan_obj.processed_scan
        sum_residuals = scan["Residual"].sum()
        sum_square_residuals = (scan["Residual"] ** 2).sum()
        info = pd.DataFrame(
            {
                "Value": [self.a, self.c, self.s, sum_residuals, sum_square_residuals],
                "Error": [self.a_err, self.c_err, self.s_err, np.nan, np.nan],
            },
            index=["a", "c", "s", "Sum(Residuals)", "Sum(Residuals^2)"],
        )
        return info

    @property
    def residual_analysis_by_sum(self) -> pd.DataFrame:
        scan = self.scan_obj.processed_scan
        bottom = scan[scan["Interp z"] < self.c]
        top = scan[scan["Interp z"] > self.c]
        residual_symmetry_by_sum = [
            integrate.trapezoid(bottom["Residual"], bottom["Interp z"]),
            integrate.trapezoid(top["Residual"], top["Interp z"]),
        ]
        squared_residual_symmetry_by_sum = [
            integrate.trapezoid(bottom["Residual"] ** 2, bottom["Interp z"]),
            integrate.trapezoid(top["Residual"] ** 2, top["Interp z"]),
        ]
        residual_analysis = pd.DataFrame(
            {
                "Bottom": residual_symmetry_by_sum,
                "Top": squared_residual_symmetry_by_sum,
            },
            index=["Residual Symmetry", "(Residual)^2 Symmetry"],
        )
        residual_analysis["Asymmetry"] = (
            residual_analysis["Bottom"] - residual_analysis["Top"]
        )
        return residual_analysis

    @property
    def residual_symmetry(self) -> pd.DataFrame:
        scan = self.scan_obj.processed_scan
        bottom = scan[scan["Interp z"] < self.c]
        top = scan[self.scan["Interp z"] > self.c]
        bottom = (
            bottom.copy()
            .sort_values(by=["Interp z"], ascending=False)
            .reset_index(drop=True)
        )
        top = top.copy().reset_index(drop=True)
        residual_symmetry = pd.DataFrame()
        residual_symmetry["Distance from Center (mm)"] = (
            (top["Interp z"] - self.c) + (bottom["Interp z"] - self.c)
        ) / 2
        residual_symmetry["Difference in Residuals"] = (
            top["Residual"] - bottom["Residual"]
        )
        residual_symmetry.dropna(inplace=True)
        return residual_symmetry
