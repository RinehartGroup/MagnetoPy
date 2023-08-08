from __future__ import annotations
import json
from pathlib import Path
import re
from typing import Any, Literal

import pandas as pd
import matplotlib.pyplot as plt

from magnetopy.data_files import DatFile, plot_raw, plot_raw_residual
from magnetopy.experiments.plot_utils import (
    get_ylabel,
    handle_kwargs,
    handle_options,
)
from magnetopy.experiments.utils import (
    scale_dc_data,
    add_uncorrected_moment_columns,
    auto_detect_temperature,
    num_digits_after_decimal,
)
from magnetopy.parsing_utils import find_sequence_starts, label_clusters, unique_values
from magnetopy.plot_utils import default_colors, force_aspect, linear_color_gradient


class MvsH:
    class TemperatureNotInDataError(Exception):
        pass

    class SegmentError(Exception):
        pass

    class FieldCorrectionError(Exception):
        pass

    def __init__(
        self,
        dat_file: str | Path | DatFile,
        temperature: int | float | None = None,
        parse_raw: bool = False,
        **kwargs,
    ) -> None:
        if not isinstance(dat_file, DatFile):
            dat_file = DatFile(Path(dat_file), parse_raw)
        self.origin_file = dat_file.local_path.name

        # optional arguments used for algorithmic separation of
        # data at the requested temperature
        n_digits = num_digits_after_decimal(temperature) if temperature else 0
        options = {"eps": 0.001, "min_samples": 10, "n_digits": n_digits}
        options.update(kwargs)

        if temperature is None:
            temperature = auto_detect_temperature(
                dat_file, options["eps"], options["min_samples"], options["n_digits"]
            )

        self.temperature = temperature
        if dat_file.comments:
            self.data = self._set_data_from_comments(dat_file)
        else:
            self.data = self._set_data_auto(
                dat_file, options["eps"], options["min_samples"], options["n_digits"]
            )
        add_uncorrected_moment_columns(self)
        self.field_correction_file = ""
        self.scaling: list[str] = []
        self.field_range = self._determine_field_range()
        self._field_fluctuation_tolerance = 1

    def __str__(self) -> str:
        return f"MvsH at {self.temperature} K"

    def __repr__(self) -> str:
        return f"MvsH at {self.temperature} K"

    def _set_data_from_comments(self, dat_file: DatFile) -> pd.DataFrame:
        start_idx: int | None = None
        end_idx: int | None = None
        for comment_idx, (data_idx, comment_list) in enumerate(
            dat_file.comments.items()
        ):
            # ignore other experiments
            if "mvsh" not in map(str.lower, comment_list):
                continue
            # one of the comments should be a number denoting the temperature
            # may also include a unit, e.g. "300 K"
            for comment in comment_list:
                if match := re.search(r"\d+", comment):
                    found_temp = float(match.group())
                    # check to see if the unit is C otherwise assume K
                    if "C" in comment:
                        found_temp += 273
                    if found_temp == self.temperature:
                        start_idx = (
                            data_idx + 1
                        )  # +1 to skip the line containing the comment
                        end_idx = (
                            list(dat_file.comments.keys())[comment_idx + 1]
                            if comment_idx + 1 < len(dat_file.comments)
                            else (len(dat_file.data))
                        )
                        break
            if start_idx is not None:
                break
        else:
            raise self.TemperatureNotInDataError(
                f"Temperature {self.temperature} not in data in {dat_file}. "
                "Or the comments are not formatted correctly."
            )
        df = dat_file.data.iloc[start_idx:end_idx].reset_index(drop=True)
        return df

    def _set_data_auto(
        self, dat_file: DatFile, eps: float, min_samples: int, ndigits: int
    ) -> pd.DataFrame:
        file_data = dat_file.data.copy()
        file_data["cluster"] = label_clusters(
            file_data["Temperature (K)"], eps, min_samples
        )
        temps = unique_values(file_data["Temperature (K)"], eps, min_samples, ndigits)
        if self.temperature not in temps:
            raise self.TemperatureNotInDataError(
                f"Temperature {self.temperature} not in data in {dat_file}."
            )
        temperature_index = temps.index(self.temperature)
        cluster = file_data["cluster"].unique()[temperature_index]
        df = (
            file_data[file_data["cluster"] == cluster]
            .drop(columns=["cluster"])
            .reset_index(drop=True)
        )
        file_data.drop(columns=["cluster"], inplace=True)
        return df

    def simplified_data(self, sequence: str = "") -> pd.DataFrame:
        full_df = self._select_sequence(sequence) if sequence else self.data.copy()
        df = pd.DataFrame()
        df["time"] = full_df["Time Stamp (sec)"]
        df["temperature"] = full_df["Temperature (K)"]
        if self.field_correction_file:
            df["field"] = full_df["true_field"]
        else:
            df["field"] = full_df["Magnetic Field (Oe)"]
        if self.scaling:
            df["moment"] = full_df["moment"]
            df["moment_err"] = full_df["moment_err"]
            df["chi"] = full_df["chi"]
            df["chi_err"] = full_df["chi_err"]
            df["chi_t"] = full_df["chi_t"]
            df["chi_t_err"] = full_df["chi_t_err"]
        else:
            df["moment"] = full_df["uncorrected_moment"]
            df["moment_err"] = full_df["uncorrected_moment_err"]
            df["chi"] = df["moment"] / df["field"]
            df["chi_err"] = df["moment_err"] / df["field"]
            df["chi_t"] = df["chi"] * df["temperature"]
            df["chi_t_err"] = df["chi_err"] * df["temperature"]
        return df

    def scale_moment(
        self,
        mass: float = 0,
        eicosane_mass: float = 0,
        molecular_weight: float = 0,
        diamagnetic_correction: float = 0,
    ) -> None:
        scale_dc_data(
            self,
            mass,
            eicosane_mass,
            molecular_weight,
            diamagnetic_correction,
        )

    def correct_field(self, field_correction_file: str | Path) -> None:
        pd_mvsh = TrueFieldCorrection(field_correction_file)
        if len(pd_mvsh.data) != len(self.data):
            raise self.FieldCorrectionError(
                "The given Pd standard sequence does not have the same number of data "
                "points as the MvsH sequence."
            )
        self.field_correction_file = pd_mvsh.origin_file
        self.data["true_field"] = pd_mvsh.data["true_field"]
        self.field_range = self._determine_field_range()

    def _determine_field_range(self) -> tuple[float, float]:
        simplified_data = self.simplified_data()
        return simplified_data["field"].min(), simplified_data["field"].max()

    @property
    def virgin(self) -> pd.DataFrame:
        return self._select_sequence("virgin")

    @property
    def forward(self) -> pd.DataFrame:
        return self._select_sequence("forward")

    @property
    def reverse(self) -> pd.DataFrame:
        return self._select_sequence("reverse")

    @property
    def loop(self) -> pd.DataFrame:
        return self._select_sequence("loop")

    def _select_sequence(self, sequence: str) -> pd.DataFrame:
        sequence_starts = find_sequence_starts(
            self.data["Magnetic Field (Oe)"], self._field_fluctuation_tolerance
        )
        df = self.data.copy()
        segment = None
        if len(sequence_starts) == 3:
            # assume virgin -> reverse -> forward
            if sequence == "virgin":
                segment = df[sequence_starts[0] : sequence_starts[1]].reset_index(
                    drop=True
                )
            elif sequence == "reverse":
                segment = df[sequence_starts[1] - 1 : sequence_starts[2]].reset_index(
                    drop=True
                )
            elif sequence == "forward":
                segment = df[sequence_starts[2] - 1 :].reset_index(drop=True)
            elif sequence == "loop":
                segment = df[sequence_starts[1] - 1 :].reset_index(drop=True)
        elif len(sequence_starts) == 2:
            if sequence == "loop":
                segment = df
            # check to see if it's forward -> reverse or reverse -> forward
            elif (
                df.at[sequence_starts[0], "Magnetic Field (Oe)"]
                > df.at[sequence_starts[1], "Magnetic Field (Oe)"]
            ):
                if sequence == "reverse":
                    segment = df[sequence_starts[0] : sequence_starts[1]].reset_index(
                        drop=True
                    )
                elif sequence == "forward":
                    segment = df[sequence_starts[1] - 1 :].reset_index(drop=True)
            else:
                if sequence == "forward":
                    segment = df[sequence_starts[0] : sequence_starts[1]].reset_index(
                        drop=True
                    )
                elif sequence == "reverse":
                    segment = df[sequence_starts[1] - 1 :].reset_index(drop=True)
        elif len(sequence_starts) == 1:
            if sequence == "loop":
                raise self.SegmentError(
                    "Full loop requested but only one segment found"
                )
            elif sequence == "virgin":
                if abs(df.at[0, "Magnetic Field (Oe)"]) > 5:
                    raise self.SegmentError(
                        "Virgin scan requested but data does not start at zero field"
                    )
                segment = df
            elif sequence == "forward":
                if df.at[0, "Magnetic Field (Oe)"] > 0:
                    raise self.SegmentError(
                        "Forward scan requested but start field is greater than end field."
                    )
                segment = df
            elif sequence == "reverse":
                if df.at[0, "Magnetic Field (Oe)"] < 0:
                    raise self.SegmentError(
                        "Reverse scan requested but start field is less than end field."
                    )
                segment = df
        else:
            raise self.SegmentError(
                f"Something went wrong. {len(sequence_starts)} segments found"
            )
        if segment is None:
            raise self.SegmentError(f"Sequence {sequence} not found in data")
        return segment

    def plot(self, *args, **kwargs) -> tuple[plt.Figure, plt.Axes]:
        return plot_mvsh(self, *args, **kwargs)

    def plot_raw(
        self,
        segment: Literal["virgin", "forward", "reverse"] = "forward",
        **kwargs,
    ) -> tuple[plt.Figure, plt.Axes]:
        return plot_raw(self._select_sequence(segment), **kwargs)

    def plot_raw_residual(
        self,
        segment: Literal["virgin", "forward", "reverse"] = "forward",
        **kwargs,
    ) -> tuple[plt.Figure, plt.Axes]:
        return plot_raw_residual(self._select_sequence(segment), **kwargs)

    def as_dict(self) -> dict[str, Any]:
        return {
            "origin_file": self.origin_file,
            "temperature": self.temperature,
            "field_range": self.field_range,
            "field_correction_file": self.field_correction_file,
            "scaling": self.scaling,
        }

    @classmethod
    def get_all_in_file(
        cls,
        dat_file: str | Path | DatFile,
        eps: float = 0.001,
        min_samples: int = 10,
        ndigits: int = 0,
        parse_raw: bool = False,
    ) -> list[MvsH]:
        if not isinstance(dat_file, DatFile):
            dat_file = DatFile(Path(dat_file), parse_raw)
        if dat_file.comments:
            mvsh_objs = cls._get_all_mvsh_in_commented_file(dat_file)
        else:
            mvsh_objs = cls._get_all_mvsh_in_uncommented_file(
                dat_file,
                eps,
                min_samples,
                ndigits,
            )
        mvsh_objs.sort(key=lambda x: x.temperature)
        return mvsh_objs

    @classmethod
    def _get_all_mvsh_in_commented_file(cls, dat_file: DatFile) -> list[MvsH]:
        mvsh_objs = []
        for comment_list in dat_file.comments.values():
            # ignore other experiments
            if "mvsh" not in map(str.lower, comment_list):
                continue
            # one of the comments should be a number denoting the temperature
            # may also include a unit, e.g. "300 K"
            for comment in comment_list:
                if match := re.search(r"\d+", comment):
                    temp = float(match.group())
                    # check to see if the unit is C otherwise assume K
                    if "C" in comment:
                        temp += 273
                    mvsh_objs.append(cls(dat_file, temp))
        return mvsh_objs

    @classmethod
    def _get_all_mvsh_in_uncommented_file(
        cls,
        dat_file: DatFile,
        eps: float,
        min_samples: int,
        ndigits: int,
    ) -> list[MvsH]:
        file_data = dat_file.data
        file_data["cluster"] = label_clusters(
            file_data["Temperature (K)"], eps, min_samples
        )
        temps = unique_values(file_data["Temperature (K)"], eps, min_samples, ndigits)
        mvsh_objs = []
        for temp in temps:
            mvsh_objs.append(cls(dat_file, temp, eps=eps, min_samples=min_samples))
        return mvsh_objs


class TrueFieldCorrection(MvsH):
    """
    Corrects magnetic field for flux trapping according to
    https://qdusa.com/siteDocs/appNotes/1500-021.pdf
    """

    def __init__(self, sequence: str | Path):
        dat_file = self._get_dat_file(sequence)
        super().__init__(dat_file)
        self.pd_mass = self._get_mass(dat_file)  # mass of the Pd standard in mg
        self._add_true_field()

    def _get_dat_file(self, sequence: str) -> DatFile:
        if Path(sequence).is_file():
            return DatFile(sequence)
        mp_cal = Path().home() / ".magnetopy/calibration"
        if (Path(sequence).suffix == ".dat") and (
            mp_cal / "calibration_files" / sequence
        ).is_file():
            return DatFile(mp_cal / "calibration_files" / sequence)
        with open(mp_cal / "calibration.json", "r", encoding="utf-8") as f:
            cal_json = json.load(f)
        if sequence in cal_json["mvsh"]:
            seq_dat = cal_json["mvsh"][sequence]
            return DatFile(mp_cal / "calibration_files" / seq_dat)
        raise FileNotFoundError(
            f"Could not find the requested sequence: {sequence}. "
            "TrueFieldCorrection requires either the name of a sequence listed in "
            f"{mp_cal / 'calibration.json'}, the name of a .dat file in "
            f"{mp_cal / 'calibration_files'}, or the path to a .dat file."
        )

    @staticmethod
    def _get_mass(dat_file: DatFile) -> float:
        for line in dat_file.header:
            category = line[0]
            if category != "INFO":
                continue
            info = line[2]
            if info == "SAMPLE_MASS":
                return float(line[1])
        raise ValueError("Could not find the sample mass in the .dat file header.")

    def _add_true_field(self):
        chi_g = 5.25e-6  # emu Oe / g
        self.data["true_field"] = self.data["uncorrected_moment"] / (
            chi_g * self.pd_mass * 1e-3
        )


def plot_mvsh(
    mvsh: MvsH | list[MvsH],
    normalized: bool = False,
    sequence: str = "",
    colors: str | list[str] = "auto",
    labels: str | list[str] | None = "auto",
    title: str = "",
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    if isinstance(mvsh, list) and len(mvsh) == 1:
        mvsh = mvsh[0]
    if isinstance(mvsh, MvsH):
        if isinstance(colors, list) or isinstance(labels, list):
            raise ValueError(
                "If plotting a single MvsH, `colors` and `labels` must be a single value"
            )
        return plot_single_mvsh(
            mvsh=mvsh,
            normalized=normalized,
            sequence=sequence,
            color=colors,
            label=labels,
            title=title,
            **kwargs,
        )
    if colors != "auto" and not isinstance(colors, list):
        raise ValueError(
            "If plotting multiple MvsH, `colors` must be a list or 'auto'."
        )
    if labels is not None and labels != "auto" and not isinstance(labels, list):
        raise ValueError(
            "If plotting multiple MvsH, `labels` must be a list or 'auto' or `None`."
        )
    return plot_multiple_mvsh(
        mvsh,
        normalized=normalized,
        sequence=sequence,
        colors=colors,
        labels=labels,
        title=title,
        **kwargs,
    )


def plot_single_mvsh(
    mvsh: MvsH,
    normalized: bool = False,
    sequence: str = "",
    color: str = "black",
    label: str | None = "auto",
    title: str = "",
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    options = handle_kwargs(**kwargs)

    fig, ax = plt.subplots()
    x = mvsh.simplified_data(sequence)["field"] / 10000
    y = mvsh.simplified_data(sequence)["moment"]
    y = y / y.max() if normalized else y
    if label is None:
        ax.plot(x, y, c=color)
    else:
        if label == "auto":
            label = f"{mvsh.temperature} K"
        ax.plot(x, y, c=color, label=label)

    ax.set_xlabel("Field (T)")
    if normalized:
        ax.set_ylabel("Normalized Magnetization")
    else:
        ylabel = get_ylabel("moment", mvsh.scaling)
        ax.set_ylabel(ylabel)

    handle_options(ax, label, title, options)

    force_aspect(ax)
    if options["save"]:
        plt.savefig(
            options["save"], dpi=300, bbox_inches="tight", facecolor="w", edgecolor="w"
        )
    return fig, ax


def plot_multiple_mvsh(
    mvsh: list[MvsH],
    normalized: bool = False,
    sequence: str = "",
    colors: list[str] | Literal["auto"] = "auto",
    labels: list[str] | None = None,
    title: str = "",
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    options = handle_kwargs(**kwargs)

    if colors == "auto":
        colors = default_colors(len(mvsh))
    if _check_if_variable_temperature(mvsh):
        mvsh.sort(key=lambda x: x.temperature)
        colors = linear_color_gradient("blue", "red", len(mvsh))
        if labels == "auto":
            labels = [f"{x.temperature} K" for x in mvsh]
    if labels is None:
        labels: list[None] = [None] * len(mvsh)

    fig, ax = plt.subplots()
    for m, color, label in zip(mvsh, colors, labels):
        x = m.simplified_data(sequence)["field"] / 10000
        y = m.simplified_data(sequence)["moment"]
        y = y / y.max() if normalized else y
        if label:
            ax.plot(x, y, c=color, label=label)
        else:
            ax.plot(x, y, c=color)

    ax.set_xlabel("Field (T)")
    if normalized:
        ax.set_ylabel("Normalized Magnetization")
    else:
        ylabel = get_ylabel("moment", mvsh[0].scaling)
        ax.set_ylabel(ylabel)

    handle_options(ax, labels[0], title, options)
    force_aspect(ax)
    if options["save"]:
        plt.savefig(
            options["save"], dpi=300, bbox_inches="tight", facecolor="w", edgecolor="w"
        )
    return fig, ax


def _check_if_variable_temperature(mvsh: list[MvsH]):
    first_temp = mvsh[0].temperature
    for mvsh_obj in mvsh:
        if mvsh_obj.temperature != first_temp:
            return True
    return False
