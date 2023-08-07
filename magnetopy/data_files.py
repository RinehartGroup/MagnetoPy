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

from magnetopy.plot_helpers import linear_color_gradient, force_aspect


class FileNameWarning(UserWarning):
    pass


class NoRawDataError(Exception):
    pass


class GenericFile:
    """A class containing basic metadata about a file.

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

    Methods
    -------
    as_dict()
        Serializes the GenericFile object to a dictionary.
    """

    def __init__(self, file_path: str | Path, experiment_type: str = "") -> None:
        """A class containing basic metadata about a file.

        Parameters
        ----------
        file_path : str | Path
            The path to the file.
        experiment_type : str, optional
            The type of experiment the file is associated with, by default "".
        """
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

    Attributes
    ----------
    local_path : Path
        The path to the .dat file.
    header : list[list[str]]
        The header of the .dat file.
    data : pd.DataFrame
        The data from the .dat file.
    comments : OrderedDict[str, list[str]]
        Any comments found within the "[Data]" section of the .dat file.
    length : int
        The length of the .dat file in bytes.
    sha512 : str
        The SHA512 hash of the .dat file.
    date_created : datetime
        The date and time the .dat file was created.
    experiments_in_file : list[str]
        The experiments contained in the .dat file. Can include "mvsh", "zfc", "fc",
        and/or "zfcfc".

    Methods
    -------
    as_dict()
        Serializes the DatFile object to a dictionary.
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
        raw_dat = DatFile(rw_dat_file)
        raw_scans = create_raw_scans(raw_dat)
        self.combine_dat_and_raw_dfs(raw_scans)

    def combine_dat_and_raw_dfs(self, raw: list[DcMeasurement]) -> None:
        # the .rw.dat file does not account for comments in the .dat file
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

    def plot_raw(self, *args, **kwargs):
        return plot_raw(self, *args, **kwargs)

    def plot_raw_residual(self, *args, **kwargs):
        return plot_raw_residual(self, *args, **kwargs)

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


def filename_label(filename: str, experiment: str, suppress_warnings: bool) -> str:
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


class ScanHeader:
    def __init__(self, direction: str, up_header: pd.Series) -> None:
        self.text: str = up_header["Comment"]
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

    def _get_value(self, regex: str) -> float:
        return float(re.search(regex, self.text).group(1))

    @property
    def avg_field(self):
        return (self.low_field + self.high_field) / 2

    def __repr__(self):
        return f"ScanHeader({self.direction}, {self.avg_field:.2f} Oe, {self.avg_temp:.2f} K)"

    def __str__(self):
        return f"{self.direction} scan at {self.avg_field:.2f} Oe, {self.avg_temp:2f} K"


class RawScan:
    def __init__(self, direction: str, scan: pd.DataFrame) -> None:
        self.direction = direction
        self.data = scan.copy()
        self.data.drop(
            columns=["Comment", "Fixed C Fitted (V)", "Free C Fitted (V)"], inplace=True
        )
        self.data.reset_index(drop=True, inplace=True)
        self.start_time = self.data["Time Stamp (sec)"].iloc[0]

    def __repr__(self):
        return f"RawScan({self.direction} at {self.start_time} sec)"

    def __str__(self):
        return f"RawScan({self.direction} at {self.start_time} sec)"


class ProcessedScan:
    def __init__(self, scan: pd.DataFrame) -> None:
        self.data = scan.copy()
        self.data.drop(
            columns=["Comment", "Raw Voltage (V)", "Processed Voltage (V)"],
            inplace=True,
        )
        self.data.reset_index(drop=True, inplace=True)
        self.start_time = self.data["Time Stamp (sec)"].iloc[0]

    def __repr__(self):
        return f"ProcessedScan({self.start_time} sec)"

    def __str__(self):
        return f"ProcessedScan({self.start_time} sec)"


class DcMeasurement:
    """
    [Quantum Design app note](https://www.qdusa.com/siteDocs/appNotes/1500-022.pdf)
    """

    def __init__(
        self,
        up_header: pd.Series,
        up_scan: pd.DataFrame,
        down_header: pd.Series,
        down_scan: pd.DataFrame,
        processed_scan: pd.DataFrame,
    ) -> None:
        self.up_header = ScanHeader("up", up_header)
        self.up_scan = RawScan("up", up_scan)
        self.down_header = ScanHeader("down", down_header)
        self.down_scan = RawScan("down", down_scan)
        self.processed_scan = ProcessedScan(processed_scan)

    def __repr__(self):
        return f"DcMeasurement({self.up_header.avg_field:.2f} Oe, {self.up_header.avg_temp:.2f} K)"

    def __str__(self):
        return f"DcMeasurement({self.up_header.avg_field:.2f} Oe, {self.up_header.avg_temp:.2f} K)"


def create_raw_scans(raw_dat: DatFile) -> list[DcMeasurement]:
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
            processed_scan = df.iloc[
                raw_idx[i + 1] + (raw_idx[i + 1] - raw_idx[i]) : raw_idx[i + 2]
            ]
        except IndexError:
            processed_scan = df.iloc[raw_idx[i + 1] + (raw_idx[i + 1] - raw_idx[i]) :]
        scans.append(
            DcMeasurement(up_header, up_scan, down_header, down_scan, processed_scan)
        )
    return scans


def plot_raw(
    dat_file: DatFile,
    data_slice: tuple[int, int] | None = None,
    scan: Literal[
        "up",
        "up_raw",
        "down",
        "down_raw",
        "processed",
    ] = "up",
    center: Literal[
        "free",
        "fixed",
    ] = "free",
    colors: tuple[str, str] | None = None,
    label: bool = True,
    title: str = "",
):
    if "raw_scan" not in dat_file.data.columns:
        raise NoRawDataError("This DatFile object does not contain raw data.")
    data = dat_file.data.drop(columns=["Comment"])
    if data_slice is not None:
        data = data.iloc[slice(*data_slice)]
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
        ax.legend()
    if title:
        ax.set_title(title)
    force_aspect(ax)
    return fig, ax


def plot_raw_residual(
    dat_file: DatFile,
    data_slice: tuple[int, int] | None = None,
    scan: Literal["up", "down"] = "up",
    center: Literal["free", "fixed"] = "free",
    colors: tuple[str, str] | None = None,
    label: bool = True,
    title: str = "",
):
    if "raw_scan" not in dat_file.data.columns:
        raise NoRawDataError("This DatFile object does not contain raw data.")
    data = dat_file.data.drop(columns=["Comment"])
    if data_slice is not None:
        data = data.iloc[slice(*data_slice)]
    start_label, end_label = _get_voltage_scan_labels(data)

    scan_objs: list[DcMeasurement] = data["raw_scan"]
    scans_w_squid_range = _get_selected_scans(scan, scan_objs)
    processed_scans = [scan_obj.processed_scan.data for scan_obj in scan_objs]

    if colors is None:
        colors = ("purple", "orange")
    colors = linear_color_gradient(colors[0], colors[1], len(scans_w_squid_range))

    fig, ax = plt.subplots()
    for i, ((scan_df, squid_range), processed_df, color) in enumerate(
        zip(scans_w_squid_range, processed_scans, colors)
    ):
        row_label = None
        if label and i == 0:
            row_label = start_label
        elif label and i == len(scans_w_squid_range) - 1:
            row_label = end_label

        x = scan_df["Raw Position (mm)"]
        if center == "free":
            y_processed = processed_df["Free C Fitted (V)"] * squid_range
        else:
            y_processed = processed_df["Fixed C Fitted (V)"] * squid_range
        y_raw = scan_df["Processed Voltage (V)"] * squid_range
        y = y_raw - y_processed

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
            (scan_obj.up_scan.data, scan_obj.up_header.squid_range)
            for scan_obj in scan_objs
        ]
    elif scan in ["down", "down_raw"]:
        selected_scans: tuple[pd.DataFrame, int] = [
            (scan_obj.down_scan.data, scan_obj.down_header.squid_range)
            for scan_obj in scan_objs
        ]
    else:
        selected_scans: tuple[pd.DataFrame, int] = [
            (scan_obj.processed_scan.data, scan_obj.up_header.squid_range)
            for scan_obj in scan_objs
        ]
    return selected_scans
