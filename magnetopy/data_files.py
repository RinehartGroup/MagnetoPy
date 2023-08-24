from __future__ import annotations
import csv
import hashlib
from pathlib import Path
import re
from typing import Any, Literal
from collections import OrderedDict
from datetime import datetime
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from magnetopy.plot_utils import linear_color_gradient, force_aspect


class FileNameWarning(UserWarning):
    pass


class NoRawDataError(Exception):
    pass


class GenericFile:
    """A class containing basic metadata about a file.

    Parameters
    ----------
    file_path : str | Path
        The path to the file.
    experiment_type : str, optional
        The type of experiment the file is associated with, by default "".

    Attributes
    ----------
    local_path : Path
        The path to the file.
    length : int
        The length of the file in bytes.
    sha512 : str
        The SHA512 hash of the file.
    date_created : datetime
        The date and time the file was created.
    experiment_type : str
        The type of experiment the file is associated with.
    """

    def __init__(self, file_path: str | Path, experiment_type: str = "") -> None:
        self.local_path = Path(file_path)
        self.length = self.local_path.stat().st_size
        self.date_created = datetime.fromtimestamp(self.local_path.stat().st_ctime)
        self.sha512 = self._determine_sha512()
        self.experiment_type = experiment_type

    def __str__(self) -> str:
        return f"GenericFile({self.local_path.name})"

    def __repr__(self) -> str:
        return f"GenericFile({self.local_path.name})"

    def _determine_sha512(self) -> str:
        buf_size = 4 * 1024 * 1024  # 4MB chunks
        hasher = hashlib.sha512()
        with self.local_path.open("rb") as f:
            while data := f.read(buf_size):
                hasher.update(data)
        return hasher.hexdigest()

    def as_dict(self) -> dict[str, Any]:
        """Serializes the GenericFile object to a dictionary.

        Returns
        -------
        dict[str, Any]
            Contains the following keys: local_path, length, date_created, sha512
        """
        return {
            "experiment_type": self.experiment_type,
            "local_path": str(self.local_path),
            "length": self.length,
            "date_created": self.date_created.isoformat(),
            "sha512": self.sha512,
        }


class DatFile(GenericFile):
    """A class for reading and storing data from a Quantum Design .dat file from a
    MPMS3 magnetometer.

    Parameters
    ----------
    file_path : str | Path
        The path to the .dat file.
    parse_raw : bool, optional
        By default `False`. If `True` and there is a corresponding .rw.dat file, the
        raw data will be parsed and stored in the `raw_scan` column of the `data`
        attribute.


    Attributes
    ----------
    local_path : Path
        The path to the .dat file.
    header : list[list[str]]
        The header of the .dat file.
    data : pd.DataFrame
        The data from the .dat file.
    comments : OrderedDict[str, list[str]]
        Any comments found in the "Comment" column within the "[Data]" section of the
        .dat file.
    length : int
        The length of the .dat file in bytes.
    sha512 : str
        The SHA512 hash of the .dat file.
    date_created : datetime
        The date and time the .dat file was created.
    experiments_in_file : list[str]
        The experiments contained in the .dat file. Can include "mvsh", "zfc", "fc",
        and/or "zfcfc".
    """

    def __init__(self, file_path: str | Path, parse_raw: bool = False) -> None:
        super().__init__(file_path, "magnetometry")
        self.header = self._read_header()
        self.data = self._read_data()
        self.comments = self._get_comments()
        self.date_created = self._get_date_created()
        self.experiments_in_file = self._get_experiments_in_file()
        if parse_raw:
            rw_dat_file = self.local_path.parent / (self.local_path.stem + ".rw.dat")
            if rw_dat_file.exists():
                self.append_raw_data(rw_dat_file)

    def __str__(self) -> str:
        return f"DatFile({self.local_path.name})"

    def __repr__(self) -> str:
        return f"DatFile({self.local_path.name})"

    def _read_header(self, delimiter: str = "\t") -> list[list[str]]:
        header: list[list[str]] = []
        with self.local_path.open(encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=delimiter)
            for row in reader:
                header.append(row)
                if row[0] == "[Data]":
                    break
        if len(header[2]) == 1:
            # some .dat files have a header that is delimited by commas
            header = self._read_header(delimiter=",")
        return header

    def _read_data(
        self,
        sep: str = "\t",
    ) -> pd.DataFrame:
        skip_rows = len(self.header)
        df = pd.read_csv(self.local_path, sep=sep, skiprows=skip_rows)
        if df.shape[1] == 1:
            # some .dat files have a header that is delimited by commas
            df = self._read_data(sep=",")
        return df

    def _get_comments(self) -> OrderedDict[str, list[str]]:
        comments = self.data["Comment"].dropna()
        comments = OrderedDict(comments)
        for key, value in comments.items():
            comments[key] = [comment.strip() for comment in value.split(",")]
        return comments

    def _get_date_created(self) -> datetime:
        for line in self.header:
            if line[0] == "FILEOPENTIME":
                day = line[2]
                hour = line[3]
                break
        hour24 = datetime.strptime(hour, "%I:%M %p")
        day = [int(x) for x in day.split("/")]
        return datetime(day[2], day[0], day[1], hour24.hour, hour24.minute)

    def _get_experiments_in_file(self) -> list[str]:
        experiments = []
        if self.comments:
            for comments in self.comments.values():
                for comment in comments:
                    if comment.lower() in ["mvsh", "zfc", "fc", "zfcfc"]:
                        experiments.append(comment.lower())
        elif (filename := filename_label(self.local_path.name, "", True)) != "unknown":
            experiments.append(filename)
        else:
            if len(self.data["Magnetic Field (Oe)"].unique()) == 1:
                experiments.append("zfcfc")
            else:
                experiments.append("mvsh")
        return experiments

    def append_raw_data(self, rw_dat_file: str | Path) -> None:
        """Adds a column "raw_scan" to the `data` attribute containing the raw data
        from the .rw.dat file.

        Parameters
        ----------
        rw_dat_file : str | Path
            The path to the .rw.dat file.
        """
        raw_scans = create_raw_scans(rw_dat_file)
        self.combine_dat_and_raw_dfs(raw_scans)

    def combine_dat_and_raw_dfs(self, raw: list[DcMeasurement]) -> None:
        """Data from the .rw.dat file is converted to a list of DcMeasurement objects
        which must be integrated with the `DataFrame` stored in the `data` attribute.
        This is not completely straightforward in cases where there are comments in
        the .dat file. This method takes the list of DcMeasurement objects and
        integrates them with the `DataFrame` stored in the `data` attribute.


        Parameters
        ----------
        raw : list[DcMeasurement]
            A list of DcMeasurement objects created from the .rw.dat file.
        """
        if len(self.data) == len(raw):
            # there are no comments in the .dat file
            self.data["raw_scan"] = raw
        else:
            # we need to skip rows that have comments
            has_comment = self.data["Comment"].notna()
            new_raw = []
            j = 0
            for i in range(len(self.data)):
                if has_comment[i]:
                    new_raw.append(np.nan)
                else:
                    new_raw.append(raw[j])
                    j += 1
            self.data["raw_scan"] = new_raw

    def plot_raw(
        self,
        data_slice: tuple[int, int] | None = None,
        scan: Literal[
            "up",
            "up_raw",
            "down",
            "down_raw",
            "fit",
        ] = "up",
        center: Literal[
            "free",
            "fixed",
        ] = "free",
        colors: tuple[str, str] = ("purple", "orange"),
        label: bool = True,
        title: str = "",
    ) -> tuple[plt.Figure, plt.Axes]:
        """If the `data` attribute contains raw data, this method will plot it.

        Parameters
        ----------
        data_slice : tuple[int, int] | None, optional
            The slice of data to plot (start, stop). `None` by default. If `None`, all
            data will be plotted.
        scan : Literal["up", "up_raw", "down", "down_raw", "fit"], optional
            Which data to plot. `"up"` and `"down"` will plot the processed directional
            scans (which have been adjusted for drift and shifted to center the waveform
            around 0, but have not been fit), `"up_raw"` and `"down_raw"` will plot the raw
            voltages as the come straight off the SQUID, and `"fit"` will plot the
            fit data (which is the result of fitting the up and down scans). `"up"` by
            default.
        center : Literal["free", "fixed"], optional
            Only used if `scan` is `"fit"`; determines whether to plot the "Free C
            Fitted" or "Fixed C Fitted" data. `"free"` by default.
        colors : tuple[str, str], optional
            The (start, end) colors for the color gradient. `"purple"` and `"orange"` by
            default.
        label : bool, optional
            Default `True`. Whether to put labels on the plot for the initial and final
            scans.
        title : str, optional
            The title of the plot. `""` by default.

        Returns
        -------
        tuple[plt.Figure, plt.Axes]
            The figure and axes objects created by `plot_raw`.

        Raises
        ------
        NoRawDataError
        """
        return plot_raw(self.data, data_slice, scan, center, colors, label, title)

    def plot_raw_residual(
        self,
        data_slice: tuple[int, int] | None = None,
        scan: Literal["up", "down"] = "up",
        center: Literal["free", "fixed"] = "free",
        colors: tuple[str, str] | None = None,
        label: bool = True,
        title: str = "",
    ) -> tuple[plt.Figure, plt.Axes]:
        """If the `data` attribute contains raw data, this method will plot the
        residual between the raw data and the fit data.

        Parameters
        ----------
        data_slice : tuple[int, int] | None, optional
            The slice of data to plot (start, stop). `None` by default. If `None`, all
            data will be plotted.
        scan : Literal["up", "down"], optional
            Which data to use in the residual calculation. `"up"` and `"down"` will use the
            processed directional scans (which have been adjusted for drift and shifted to
            center the waveform around 0, but have not been fit). `"up"` by default.
        center : Literal["free", "fixed"], optional
            Only used if `scan` is `"fit"`; determines whether to plot the "Free C
            Fitted" or "Fixed C Fitted" data. `"free"` by default.
        colors : tuple[str, str], optional
            The (start, end) colors for the color gradient. `"purple"` and `"orange"` by
            default.
        label : bool, optional
            Default `True`. Whether to put labels on the plot for the initial and final
            scans.
        title : str, optional
            The title of the plot. `""` by default.

        Returns
        -------
        tuple[plt.Figure, plt.Axes]
            The figure and axes objects created by `plot_raw_residual`.

        Raises
        ------
        NoRawDataError
        """
        return plot_raw_residual(
            self.data, data_slice, scan, center, colors, label, title
        )

    def as_dict(self) -> dict[str, Any]:
        """Serializes the DatFile object to a dictionary.

        Returns
        -------
        dict[str, Any]
            Contains the following keys: local_path, length, date_created, sha512,
            experiments_in_file.
        """
        output = super().as_dict()
        output["experiments_in_file"] = self.experiments_in_file
        return output


def filename_label(
    filename: str, experiment: str = "", suppress_warnings: bool = False
) -> str:
    """Determines the experiment type based on the filename. Currently supported
    experiments are: "zfc", "fc", "zfcfc", and "mvsh".

    Parameters
    ----------
    filename : str
        The name of the file.
    experiment : str
        Used in the case when you are looking for a specific experiment type.
    suppress_warnings : bool
        If an `experiment` is passed and the filename indicates a different
        experiment, a warning will be raised. If `suppress_warnings` is `True`, the
        warning will be suppressed. Defaults to `False`.

    Returns
    -------
    str
        The experiment type.

    Raises
    ------
    FileNameWarning
        If `experiment` is passed and the filename indicates a different experiment.
    """
    name = filename.lower()
    label = "unknown"
    if "zfcfc" in name:
        label = "zfcfc"
    elif "zfc" in name:
        label = "zfc"
    elif "fc" in name:
        label = "fc"
    elif "mvsh" in name:
        label = "mvsh"
    if (
        experiment
        and label in ["zfc", "fc"]
        and label != experiment
        and not suppress_warnings
    ):
        warnings.warn(
            (
                f"You have initialized a {experiment.upper()} object but the "
                f"file name {filename} indicates that it is {label.upper()}. "
                "You can suppress this warning by passing `suppress_warnings=True` to "
                "the constructor."
            ),
            FileNameWarning,
        )
    return label


class RawDcScan:
    """A class for storing the header information from a single raw scan.

    Parameters
    ----------
    direction : Literal["up", "down"]
        The direction of the scan.
    header : pd.Series
        The header information from the .dat file. The information is initially stored
        in the "Comment" column in a single row preceding the scan data.

    Attributes
    ----------
    text : str
        The original text from the "Comment" column.
    direction : Literal["up", "down"]
        The direction of the scan.
    low_temp : float
        The lowest temperature recorded during the combined DC scan.
    high_temp : float
        The highest temperature recorded during the combined DC scan.
    avg_temp : float
        The average temperature recorded during the combined DC scan.
    low_field : float
        The lowest magnetic field (in Oe) recorded during the combined DC scan.
    high_field : float
        The highest magnetic field (in Oe) recorded during the combined DC scan.
    drift : float
        The amount of drift (in V/S) between the DOWN->UP and UP->DOWN scans.
    slope : float
        The linear slope (in V/mm) between the DOWN->UP and UP->DOWN scans.
    squid_range : float
        The SQUID range [1, 10, 100, or 1000] used during the combined DC scan.
    given_center : float
        The center position (in mm) as set during the sample installation wizard.
    calculated_center : float
        The calculated center position (in mm) from the Free C Fitted data.
    amp_fixed : float
        The amplitude (in V) of the Fixed C Fitted data.
    amp_free : float
        The amplitude (in V) of the Free C Fitted data.
    data : pd.DataFrame
        The raw scan data. Columns are: "Time Stamp (sec)", "Raw Position (mm)",
        "Raw Voltage (V)", "Processed Voltage (V)". The Raw Voltage data from both
        up and down scans are corrected for drift and shifted to center the waveform
        around V=0, and the results of those corrections are stored in the
        "Processed Voltage (V)" column.
    start_time : float
        The time stamp (in seconds) of the first data point in the scan.
    """

    def __init__(
        self, direction: Literal["up", "down"], header: pd.Series, scan: pd.DataFrame
    ) -> None:
        self.text: str = header["Comment"]
        self.direction = direction
        self.low_temp = self._get_value(r"low temp = (\d+\.\d+) K")
        self.high_temp = self._get_value(r"high temp = (\d+\.\d+) K")
        self.avg_temp = self._get_value(r"avg. temp = (\d+\.\d+) K")
        self.low_field = self._get_value(r"low field = (-?\d+\.\d+) Oe")
        self.high_field = self._get_value(r"high field = (-?\d+\.\d+) Oe")
        self.drift = self._get_value(r"drift = (-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?) V/s")
        self.slope = self._get_value(r"slope = (-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?) V/mm")
        self.squid_range = self._get_value(r"squid range = (\d+)")
        self.given_center = self._get_value(r"given center = (\d+\.\d+) mm")
        self.calculated_center = self._get_value(r"calculated center = (\d+\.\d+) mm")
        self.amp_fixed = self._get_value(r"amp fixed = (-?\d+\.\d+) V")
        self.amp_free = self._get_value(r"amp free =(-?\d+\.\d+) V")
        self.data = scan.copy()
        self.data.drop(
            columns=["Comment", "Fixed C Fitted (V)", "Free C Fitted (V)"], inplace=True
        )
        self.data.reset_index(drop=True, inplace=True)
        self.start_time = self.data["Time Stamp (sec)"].iloc[0]

    def _get_value(self, regex: str) -> float:
        return float(re.search(regex, self.text).group(1))

    @property
    def avg_field(self):
        return (self.low_field + self.high_field) / 2

    def __repr__(self):
        return f"RawDcScan({self.direction}, {self.avg_field:.2f} Oe, {self.avg_temp:.2f} K)"

    def __str__(self):
        return f"{self.direction} scan at {self.avg_field:.2f} Oe, {self.avg_temp:2f} K"


class FitDcScan:
    """The FitDcScan class stores the simulated voltage data from fits to the
    directional scans; one for the case in which the center position is allowed to
    float (Free C Fitted) and one for the case in which the center position is fixed
    (Fixed C Fitted) based on the initial centering of the sample.

    Parameters
    ----------
    scan : pd.DataFrame
        The fit scan data from the .rw.dat file.

    Attributes
    ----------
    data : pd.DataFrame
        The fit scan data from the .rw.dat file. Columns are: "Time Stamp (sec)",
        "Raw Position (mm)", "Fixed C Fitted (V)", "Free C Fitted (V)".
    start_time : float
        The time stamp (in seconds) of the first data point in the scan.

    """

    def __init__(self, scan: pd.DataFrame) -> None:
        self.data = scan.copy()
        self.data.drop(
            columns=["Comment", "Raw Voltage (V)", "Processed Voltage (V)"],
            inplace=True,
        )
        self.data.reset_index(drop=True, inplace=True)
        self.start_time = self.data["Time Stamp (sec)"].iloc[0]

    def __repr__(self):
        return f"FitDcScan({self.start_time} sec)"

    def __str__(self):
        return f"FitDcScan({self.start_time} sec)"


class DcMeasurement:
    r"""
    The Quantum Design software fits the Processed Voltage data from the up and down
    scans and uses the fit values with system-specific calibration factors to convert
    the voltages to magnetic moment. This class stores both the raw and fit
    data from a single DC measurement.

    Parameters
    ----------
    up_header : pd.Series
        The header information from the .rw.dat file for the up scan.
    up_scan : pd.DataFrame
        The raw scan data from the .rw.dat file for the up scan.
    down_header : pd.Series
        The header information from the .rw.dat file for the down scan.
    down_scan : pd.DataFrame
        The raw scan data from the .rw.dat file for the down scan.
    fit_scan : pd.DataFrame
        The fit scan data from the .rw.dat file.

    Attributes
    ----------
    up : RawDcScan
        The information about and data from the up scan.
    down : RawDcScan
        The information about and data from the down scan.
    fit_scan : FitDcScan
        The fit scan data determined by fitting the up and down scans.

    Notes
    --------
    Information on the structure of a .rw.dat file can be found in the Quantum Design
    app note[1].

    The fit scan is determined by fitting the up and down scans to the following
    equation[1]:

    ```math
    V(z) = S + A \left\{ 2 \left[ R^2 + (z - C)^2 \right]^{-\frac{3}{2}} -
    [R^2 + (L + z - C)^2]^{-\frac{3}{2}} - [R^2 + (-L + z - C)^2]^{-\frac{3}{2}}
    \right\}
    ```


    where S is the offset voltage, A is the amplitude, R is the radius of the
    gradiometer, L is half the length of the gradiometer, and C is the sample center
    position.

    References
    ----------
    [MPMS3 Application Note 1500-022: MPMS3 .rw.dat file format](
        https://www.qdusa.com/siteDocs/appNotes/1500-022.pdf
    )

    [MPMS3 Application Note 1500-023: Background subtraction using the MPMS3](
        https://qdusa.com/siteDocs/appNotes/1500-023.pdf
    )
    """

    def __init__(
        self,
        up_header: pd.Series,
        up_scan: pd.DataFrame,
        down_header: pd.Series,
        down_scan: pd.DataFrame,
        fit_scan: pd.DataFrame,
    ) -> None:
        self.up = RawDcScan("up", up_header, up_scan)
        self.down = RawDcScan("down", down_header, down_scan)
        self.fit_scan = FitDcScan(fit_scan)

    def __repr__(self):
        return f"DcMeasurement({self.up.avg_field:.2f} Oe, {self.up.avg_temp:.2f} K)"

    def __str__(self):
        return f"DcMeasurement({self.up.avg_field:.2f} Oe, {self.up.avg_temp:.2f} K)"


def create_raw_scans(raw_dat: str | Path | DatFile) -> list[DcMeasurement]:
    """Parse a .rw.dat file and return a list of `DcMeasurement` objects.

    Parameters
    ----------
    raw_dat : str | Path | DatFile
        The path to the .rw.dat file or a `DatFile` object made from the .rw.dat file.

    Returns
    -------
    list[DcMeasurement]
        A list of `DcMeasurement` objects, where each scan is a single `DcMeasurement`.
    """
    if not isinstance(raw_dat, DatFile):
        raw_dat = DatFile(raw_dat)
    df = raw_dat.data
    raw_idx = list(raw_dat.comments.keys())
    scans = []
    for i in range(0, len(raw_idx), 2):
        up_header = df.iloc[raw_idx[i]]
        up_scan = df.iloc[raw_idx[i] + 1 : raw_idx[i + 1]]
        down_header = df.iloc[raw_idx[i + 1]]
        down_scan = df.iloc[
            raw_idx[i + 1] + 1 : raw_idx[i + 1] + (raw_idx[i + 1] - raw_idx[i])
        ]
        try:
            fit_scan = df.iloc[
                raw_idx[i + 1] + (raw_idx[i + 1] - raw_idx[i]) : raw_idx[i + 2]
            ]
        except IndexError:
            fit_scan = df.iloc[raw_idx[i + 1] + (raw_idx[i + 1] - raw_idx[i]) :]
        scans.append(
            DcMeasurement(up_header, up_scan, down_header, down_scan, fit_scan)
        )
    return scans


def plot_raw(
    data: pd.DataFrame,
    data_slice: tuple[int, int] | None = None,
    scan: Literal[
        "up",
        "up_raw",
        "down",
        "down_raw",
        "fit",
    ] = "up",
    center: Literal[
        "free",
        "fixed",
    ] = "free",
    colors: tuple[str, str] = ("purple", "orange"),
    label: bool = True,
    title: str = "",
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the raw voltage data found in the "raw_scan" column of a `DataFrame`, where
    each row contains a `DcMeasurement` object.

    Parameters
    ----------
    data : pd.DataFrame
        The `DataFrame` containing the raw data.
    data_slice : tuple[int, int] | None, optional
        The slice of data to plot (start, stop). `None` by default. If `None`, all
        data will be plotted.
    scan : Literal["up", "up_raw", "down", "down_raw", "fit"], optional
        Which data to plot. `"up"` and `"down"` will plot the processed directional
        scans (which have been adjusted for drift and shifted to center the waveform
        around 0, but have not been fit), `"up_raw"` and `"down_raw"` will plot the raw
        voltages as the come straight off the SQUID, and `"fit"` will plot the
        fit data (which is the result of fitting the up and down scans). `"up"` by
        default.
    center : Literal["free", "fixed"], optional
        Only used if `scan` is `"fit"`; determines whether to plot the "Free C
        Fitted" or "Fixed C Fitted" data. `"free"` by default.
    colors : tuple[str, str], optional
        The (start, end) colors for the color gradient. `"purple"` and `"orange"` by
        default.
    label : bool, optional
        Default `True`. Whether to put labels on the plot for the initial and final
        scans.
    title : str, optional
        The title of the plot. `""` by default.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        The figure and axes objects created by `plot_raw`.
    """
    data = _prepare_data_for_plot(data, data_slice)
    start_label, end_label = _get_voltage_scan_labels(data)

    scan_objs: list[DcMeasurement] = data["raw_scan"]
    scans_w_squid_range = _get_selected_scans(scan, scan_objs)

    if colors is None:
        colors = ("purple", "orange")
    colors = linear_color_gradient(colors[0], colors[1], len(scans_w_squid_range))

    fig, ax = plt.subplots()
    for i, ((scan_df, squid_range), color) in enumerate(
        zip(scans_w_squid_range, colors)
    ):
        row_label = None
        if label and i == 0:
            row_label = start_label
        elif label and i == len(scans_w_squid_range) - 1:
            row_label = end_label

        x = scan_df["Raw Position (mm)"]
        if scan in ["up", "down"]:
            y = scan_df["Processed Voltage (V)"] * squid_range
        elif scan in ["up_raw", "down_raw"]:
            y = scan_df["Raw Voltage (V)"] * squid_range
        else:
            if center == "free":
                y = scan_df["Free C Fitted (V)"] * squid_range
            else:
                y = scan_df["Fixed C Fitted (V)"] * squid_range
        if row_label:
            ax.plot(x, y, color=color, label=row_label)
        else:
            ax.plot(x, y, color=color)

    ax.set_xlabel("Position (mm)")
    ax.set_ylabel("Scaled Voltage (V)")
    if label:
        ax.legend(frameon=False)
    if title:
        ax.set_title(title)
    force_aspect(ax)
    return fig, ax


def plot_raw_residual(
    data: pd.DataFrame,
    data_slice: tuple[int, int] | None = None,
    scan: Literal["up", "down"] = "up",
    center: Literal["free", "fixed"] = "free",
    colors: tuple[str, str] | None = None,
    label: bool = True,
    title: str = "",
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the residual between the raw and fit voltage data found in the
    "raw_scan" column of a `DataFrame`, where each row contains a `DcMeasurement`
    object.

    Parameters
    ----------
    data : pd.DataFrame
        The `DataFrame` containing the raw data.
    data_slice : tuple[int, int] | None, optional
        The slice of data to plot (start, stop). `None` by default. If `None`, all
        data will be plotted.
    scan : Literal["up", "down"], optional
        Which data to use in the residual calculation. `"up"` and `"down"` will use the
        processed directional scans (which have been adjusted for drift and shifted to
        center the waveform around 0, but have not been fit). `"up"` by default.
    center : Literal["free", "fixed"], optional
        Determines whether to use the "Free C Fitted" or "Fixed C Fitted" data for the
        fit data. `"free"` by default.
    colors : tuple[str, str], optional
        The (start, end) colors for the color gradient. `"purple"` and `"orange"` by
        default.
    label : bool, optional
        Default `True`. Whether to put labels on the plot for the initial and final
        scans.
    title : str, optional
        The title of the plot. `""` by default.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        The figure and axes objects created by `plot_raw`.
    """
    data = _prepare_data_for_plot(data, data_slice)
    start_label, end_label = _get_voltage_scan_labels(data)

    scan_objs: list[DcMeasurement] = data["raw_scan"]
    scans_w_squid_range = _get_selected_scans(scan, scan_objs)
    fit_scans = [scan_obj.fit_scan.data for scan_obj in scan_objs]

    if colors is None:
        colors = ("purple", "orange")
    colors = linear_color_gradient(colors[0], colors[1], len(scans_w_squid_range))

    fig, ax = plt.subplots()
    for i, ((scan_df, squid_range), fit_df, color) in enumerate(
        zip(scans_w_squid_range, fit_scans, colors)
    ):
        row_label = None
        if label and i == 0:
            row_label = start_label
        elif label and i == len(scans_w_squid_range) - 1:
            row_label = end_label

        x = scan_df["Raw Position (mm)"]
        if center == "free":
            y_fit = fit_df["Free C Fitted (V)"] * squid_range
        else:
            y_fit = fit_df["Fixed C Fitted (V)"] * squid_range
        y_raw = scan_df["Processed Voltage (V)"] * squid_range
        y = y_raw - y_fit

        if row_label:
            ax.plot(x, y, color=color, label=row_label)
        else:
            ax.plot(x, y, color=color)

    ax.set_xlabel("Position (mm)")
    ax.set_ylabel("Scaled Voltage (V)")
    if label:
        ax.legend(frameon=False)
    if title:
        ax.set_title(title)
    force_aspect(ax)
    return fig, ax


def _prepare_data_for_plot(
    data: pd.DataFrame, data_slice: tuple[int, int] | None
) -> pd.DataFrame:
    data = data.copy()
    data = data[data["Comment"].isna()]
    data = data.drop(columns=["Comment"])
    if data_slice is not None:
        data = data.iloc[slice(*data_slice)]
    if "raw_scan" not in data.columns:
        raise NoRawDataError("This DatFile object does not contain raw data.")
    return data


def _get_voltage_scan_labels(data: pd.DataFrame) -> tuple[str, str]:
    start_field = data["Magnetic Field (Oe)"].iloc[0]
    end_field = data["Magnetic Field (Oe)"].iloc[-1]
    start_temp = data["Temperature (K)"].iloc[0]
    end_temp = data["Temperature (K)"].iloc[-1]
    start_label = f"{start_field:.0f} Oe, {start_temp:.0f} K"
    end_label = f"{end_field:.0f} Oe, {end_temp:.0f} K"
    return start_label, end_label


def _get_selected_scans(
    scan: str, scan_objs: list[DcMeasurement]
) -> tuple[pd.DataFrame, int]:
    if scan in ["up", "up_raw"]:
        selected_scans: tuple[pd.DataFrame, int] = [
            (scan_obj.up.data, scan_obj.up.squid_range) for scan_obj in scan_objs
        ]
    elif scan in ["down", "down_raw"]:
        selected_scans: tuple[pd.DataFrame, int] = [
            (scan_obj.down.data, scan_obj.down.squid_range) for scan_obj in scan_objs
        ]
    else:
        selected_scans: tuple[pd.DataFrame, int] = [
            (scan_obj.fit_scan.data, scan_obj.up.squid_range) for scan_obj in scan_objs
        ]
    return selected_scans
