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
    num_digits_after_decimal,
)
from magnetopy.parsing_utils import find_sequence_starts, label_clusters, unique_values
from magnetopy.plot_utils import default_colors, force_aspect, linear_color_gradient


class TemperatureDetectionError(Exception):
    pass


class MvsH:
    """A single magnetization vs. field (hysteresis) experiment at a single temperature.

    Parameters
    ----------
    dat_file : str, Path, or DatFile
        The .dat file containing the data for the experiment.
    temperature : int or float, optional
        The temperature of the experiment in Kelvin. Requied if the .dat file contains
        multiple uncommented experiments at different temperatures. If `None` and the
        .dat file contains a single experiment, the temperature will be automatically
        detected. Defaults to `None`.
    parse_raw : bool, optional
        If `True` and there is a corresponding .rw.dat file, the raw data will be
        parsed and added to the `data` attribute. Defaults to `False`.
    **kwargs : dict, optional
        Keyword arguments used for algorithmic separation of data at the requested
        temperature. See `magnetopy.parsing_utils.label_clusters` for details.

        - eps : float, optional

        - min_samples : int, optional

        - n_digits : int, optional

    Attributes
    ----------
    origin_file : str
        The name of the .dat file from which the data was parsed.
    temperature : float
        The temperature of the experiment in Kelvin.
    data : pandas.DataFrame
        The data from the experiment. Columns are taken directly from the .dat file.
    field_correction_file : str
        The name of the .dat file containing the Pd standard sequence used to correct
        the magnetic field for flux trapping. If no field correction has been applied,
        this will be an empty string.
    scaling : list of str
        The scaling applied to the data. If no scaling has been applied, this will be
        an empty list. Possible values are: `"mass"`, `"molar"`, `"eicosane"`,
        and `"diamagnetic_correction"`.
    field_range : tuple of float
        The minimum and maximum field values in the data.

    Raises
    ------
    self.TemperatureNotInDataError
        If the requested temperature is not in the data or the comments are not
        formatted correctly and the temperature cannot be automatically detected.
    self.FieldCorrectionError
        If a field correction is applied but the Pd standard sequence does not have the
        same number of data points as the MvsH sequence.
    self.SegmentError
        If the requested segment is not found in the data.
    """

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
            temperature = _auto_detect_temperature(
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

    def simplified_data(
        self, segment: Literal["", "virgin", "forward", "reverse", "loop"] = ""
    ) -> pd.DataFrame:
        """Returns a simplified version of the data, removing unnecessary columns
        and renaming the remaining columns to more convenient names.

        Parameters
        ----------
        segment : {"", "virgin", "forward", "reverse", "loop"}, optional
            Return the selected segment. By default "", which returns the full data.

        Returns
        -------
        pd.DataFrame
            The simplified data. Contains the columns:
            - `"time"` in seconds
            - `"temperature"` in Kelvin
            - `"field"` in Oe
            - `"moment"`
            - `"moment_err"`
            - `"chi"`
            - `"chi_err"`
            - `"chi_t"`
            - `"chi_t_err"`

            Where units are not specified, they are determined by the scaling applied to the
            data (see `scaling` attribute).
        """
        full_df = self.select_segment(segment) if segment else self.data.copy()
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
        """Adds the following columns to the `DataFrame` in the `data` attribute:
        `"moment"`, `"moment_err"`, `"chi"`, `"chi_err"`, `"chi_t"`, and
        `"chi_t_err"`. A record of what scaling was applied is added to the
        `scaling` attribute.

        See `magnetopy.experiments.utils.scale_dc_data` for more information.

        Parameters
        ----------
        mass : float, optional
            mg of sample, by default 0.
        eicosane_mass : float, optional
            mg of eicosane, by default 0.
        molecular_weight : float, optional
            Molecular weight of the material in g/mol, by default 0.
        diamagnetic_correction : float, optional
            Diamagnetic correction of the material in cm^3/mol, by default 0.
        """
        scale_dc_data(
            self,
            mass,
            eicosane_mass,
            molecular_weight,
            diamagnetic_correction,
        )

    def correct_field(self, field_correction_file: str | Path) -> None:
        """Applies a field correction to the data given data collected on the palladium
        standard with the same sequence as the current `MvsH` object. Adds a column
        called `"true_field"` to the `DataFrame` in the `data` attribute.

        See `magnetopy.cli.calibration_insall` for information on how to create a
        calibration directory.

        Parameters
        ----------
        field_correction_file : str | Path
            The name of the .dat file containing the Pd standard sequence, or if a
            configuration file containing calibration data is present, the name of the
            sequence in the configuration file.

        Raises
        ------
        self.FieldCorrectionError
            The true field calibration requires that the sequences of both the
            M vs. H experiment and the calibration experiment be exactly the same. This
            function only checks that they are the same length, and if they are not,
            raises this error.

        Notes
        -----
        As described in the Quantum Design application note[1], the magnetic field
        reported by the magnetometer is determined by current from the magnet power
        supply and not by direct measurement. Flux trapping in the magnet can cause
        the reported field to be different from the actual field. While always present,
        it is most obvious in hysteresis curves of soft, non-hysteretic materials. In
        some cases the forward and reverse scans can have negative and postive
        coercivities, respectively, which is not physically possible.

        The true field correction remedies this by using a Pd standard to determine the
        actual field applied to the sample. Assuming the calibration and sample
        sequences are the same, it is assumed that the flux trapping is the same for
        both sequences, and the calculated field from the measurement on the Pd
        standard is applied to the sample data.

        References
        ----------
        [1] [Correcting for the Absolute Field Error using the Pd Standard](https://qdusa.com/siteDocs/appNotes/1500-021.pdf)
        """
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
        return self.select_segment("virgin")

    @property
    def forward(self) -> pd.DataFrame:
        return self.select_segment("forward")

    @property
    def reverse(self) -> pd.DataFrame:
        return self.select_segment("reverse")

    @property
    def loop(self) -> pd.DataFrame:
        return self.select_segment("loop")

    def select_segment(
        self, segment: Literal["virgin", "forward", "reverse", "loop"]
    ) -> pd.DataFrame:
        """Returns the requested segment of the data, if it exists.

        Parameters
        ----------
        segment : {"virgin", "forward", "reverse", "loop"}
            The segment of the M vs. H data to return. "loop" refers to the combination
            of the forward and reverse scans.

        Returns
        -------
        pd.DataFrame
            The requested segment of the data.

        Raises
        ------
        self.SegmentError
            If the requested segment is not found in the data.
        """
        segment_starts = find_sequence_starts(
            self.data["Magnetic Field (Oe)"], self._field_fluctuation_tolerance
        )
        df = self.data.copy()
        requested_segment = None
        if len(segment_starts) == 3:
            # assume virgin -> reverse -> forward
            if segment == "virgin":
                requested_segment = df[
                    segment_starts[0] : segment_starts[1]
                ].reset_index(drop=True)
            elif segment == "reverse":
                requested_segment = df[
                    segment_starts[1] - 1 : segment_starts[2]
                ].reset_index(drop=True)
            elif segment == "forward":
                requested_segment = df[segment_starts[2] - 1 :].reset_index(drop=True)
            elif segment == "loop":
                requested_segment = df[segment_starts[1] - 1 :].reset_index(drop=True)
        elif len(segment_starts) == 2:
            if segment == "loop":
                requested_segment = df
            # check to see if it's forward -> reverse or reverse -> forward
            elif (
                df.at[segment_starts[0], "Magnetic Field (Oe)"]
                > df.at[segment_starts[1], "Magnetic Field (Oe)"]
            ):
                if segment == "reverse":
                    requested_segment = df[
                        segment_starts[0] : segment_starts[1]
                    ].reset_index(drop=True)
                elif segment == "forward":
                    requested_segment = df[segment_starts[1] - 1 :].reset_index(
                        drop=True
                    )
            else:
                if segment == "forward":
                    requested_segment = df[
                        segment_starts[0] : segment_starts[1]
                    ].reset_index(drop=True)
                elif segment == "reverse":
                    requested_segment = df[segment_starts[1] - 1 :].reset_index(
                        drop=True
                    )
        elif len(segment_starts) == 1:
            if segment == "loop":
                raise self.SegmentError(
                    "Full loop requested but only one segment found"
                )
            elif segment == "virgin":
                if abs(df.at[0, "Magnetic Field (Oe)"]) > 5:
                    raise self.SegmentError(
                        "Virgin scan requested but data does not start at zero field"
                    )
                requested_segment = df
            elif segment == "forward":
                if df.at[0, "Magnetic Field (Oe)"] > 0:
                    raise self.SegmentError(
                        "Forward scan requested but start field is greater than end field."
                    )
                requested_segment = df
            elif segment == "reverse":
                if df.at[0, "Magnetic Field (Oe)"] < 0:
                    raise self.SegmentError(
                        "Reverse scan requested but start field is less than end field."
                    )
                requested_segment = df
        else:
            raise self.SegmentError(
                f"Something went wrong. {len(segment_starts)} segments found"
            )
        if requested_segment is None:
            raise self.SegmentError(f"Sequence {segment} not found in data")
        return requested_segment

    def plot(
        self,
        normalized: bool = False,
        segment: str = "",
        color: str = "black",
        label: str | None = "auto",
        title: str = "",
        **kwargs,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plots the M vs. H data data.

        Parameters
        ----------
        normalized : bool, optional
            If `True`, the magnetization will be normalized to the maximum value, by
            default False.
        segment : {"", "virgin", "forward", "reverse", "loop"}, optional
            If a segment is given, only that segment will be plotted, by default "".
        color : str | list[str], optional
            The color of the plot, by default "auto". If "auto", the color will be black.
        label : str | list[str] | None, optional
            The labels to assign the `MvsH` object in the axes legend, by default "auto".
            If "auto", the label will be the `temperature` of the `MvsH` object.
        title : str, optional
            The title of the plot, by default "".
        **kwargs
            Keyword arguments mostly meant to affect the plot style. See
            `magnetopy.experiments.plot_utils.handle_options` for details.

        Returns
        -------
        tuple[plt.Figure, plt.Axes]
        """
        return plot_single_mvsh(
            self, normalized, segment, color, label, title, **kwargs
        )

    def plot_raw(
        self,
        segment: Literal["virgin", "forward", "reverse"] = "forward",
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
        colors: tuple[str, str] = ("purple", "orange"),
        label: bool = True,
        title: str = "",
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plots the raw voltage data for the requested segment.

        Parameters
        ----------
        segment : {"virgin", "forward", "reverse"}, optional
            The segment of the M vs. H data to plot, by default "forward"
        scan : Literal["up", "up_raw", "down", "down_raw", "procssed"], optional
            Which data to plot. `"up"` and `"down"` will plot the processed directional
            scans (which have been adjusted for drift and shifted to center the waveform
            around 0, but have not been fit), `"up_raw"` and `"down_raw"` will plot the raw
            voltages as the come straight off the SQUID, and `"processed"` will plot the
            processed data (which is the result of fitting the up and down scans). `"up"` by
            default.
        center : Literal["free", "fixed"], optional
            Only used if `scan` is `"processed"`; determines whether to plot the "Free C
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
        """
        return plot_raw(
            self.select_segment(segment), None, scan, center, colors, label, title
        )

    def plot_raw_residual(
        self,
        segment: Literal["virgin", "forward", "reverse"] = "forward",
        scan: Literal["up", "down"] = "up",
        center: Literal["free", "fixed"] = "free",
        colors: tuple[str, str] | None = None,
        label: bool = True,
        title: str = "",
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plots the residual of the raw voltage data for the requested segment.

        Parameters
        ----------
        segment : {"virgin", "forward", "reverse"}, optional
            The segment of the M vs. H data to plot, by default "forward"
        scan : Literal["up", "down"], optional
            Which data to use in the residual calculation. `"up"` and `"down"` will use the
            processed directional scans (which have been adjusted for drift and shifted to
            center the waveform around 0, but have not been fit). `"up"` by default.
        center : Literal["free", "fixed"], optional
            Only used if `scan` is `"processed"`; determines whether to plot the "Free C
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
        """
        return plot_raw_residual(
            self.select_segment(segment), None, scan, center, colors, label, title
        )

    def as_dict(self) -> dict[str, Any]:
        """Returns a dictionary representation of the `MvsH` object.

        Returns
        -------
        dict[str, Any]
            Keys are: `"origin_file"`, `"temperature"`, `"field_range"`,
            `"field_correction_file"`, and `"scaling"`.
        """
        return {
            "_class_": self.__class__.__name__,
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
        """Given a .dat file that contains one or more M vs. H experiments, returns a
        list of `MvsH` objects, one for each experiment.

        Parameters
        ----------
        dat_file : str | Path | DatFile
            The .dat file containing the data for the experiment.
        eps : float, optional
            See `magnetopy.parsing_utils.label_clusters` for details, by default 0.001
        min_samples : int, optional
            See `magnetopy.parsing_utils.label_clusters` for details, by default 10
        ndigits : int, optional
            See `magnetopy.parsing_utils.label_clusters` for details, by default 0
        parse_raw : bool, optional
            If `True` and there is a corresponding .rw.dat file, the raw data will be
            parsed and added to the `data` attribute. Defaults to `False`.

        Returns
        -------
        list[MvsH]
            A list of `MvsH` objects, one for each experiment in the .dat file, sorted
            by increasing temperature.
        """
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
    """A special `MvsH` class for handling the palladium standard calibration data
    used to correct the magnetic field for flux trapping. Unlikely to be used directly
    by the user, and instead will be called from the `correct_field` method of the
    `MvsH` class.

    Parameters
    ----------
    sequence : str | Path
        This could be a path to a .dat file containing the Pd standard sequence, or if
        a configuration file containing calibration data is present, the name of the
        sequence in the configuration file.

    See Also
    --------
    magnetopy.cli.calibration_install

    Notes
    -----
    As described in the Quantum Design application note[1], the magnetic field
    reported by the magnetometer is determined by current from the magnet power
    supply and not by direct measurement. Flux trapping in the magnet can cause
    the reported field to be different from the actual field. While always present,
    it is most obvious in hysteresis curves of soft, non-hysteretic materials. In
    some cases the forward and reverse scans can have negative and postive
    coercivities, respectively, which is not physically possible.

    The true field correction remedies this by using a Pd standard to determine the
    actual field applied to the sample. Provided the calibration and sample
    sequences are the same, it is assumed that the flux trapping is the same for
    both sequences, and the calculated field from the measurement on the Pd
    standard is applied to the sample data.

    References
    ----------
    [1] [Correcting for the Absolute Field Error using the Pd Standard](https://qdusa.com/siteDocs/appNotes/1500-021.pdf)
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


def _auto_detect_temperature(
    dat_file: DatFile, eps: float, min_samples: int, n_digits: int
) -> float:
    temperature: float | None = None
    if dat_file.comments:
        mvsh_comments = []
        for comment_list in dat_file.comments.values():
            if "mvsh" in map(str.lower, comment_list):
                mvsh_comments.append(comment_list)
        if len(mvsh_comments) != 1:
            raise TemperatureDetectionError(
                "Auto-parsing of MvsH objects from DatFile objects requires that "
                f"there be only one comment containing 'MvsH'. Found "
                f"{len(mvsh_comments)} comments."
            )
        comments = mvsh_comments[0]
        for comment in comments:
            if match := re.search(r"\d+", comment):
                found_temp = float(match.group())
                # check to see if the unit is C otherwise assume K
                if "C" in comment:
                    found_temp += 273
                temperature = found_temp
    else:
        temps = unique_values(
            dat_file.data["Temperature (K)"], eps, min_samples, n_digits
        )
        if len(temps) != 1:
            raise TemperatureDetectionError(
                "Auto-parsing of MvsH objects from DatFile objects requires that "
                f"there be only one temperature in the data. Found {len(temps)} "
                "temperatures."
            )
        temperature = temps[0]
    if temperature is None:
        raise TemperatureDetectionError(
            "Auto-parsing of MvsH objects from DatFile objects failed. "
            "No temperature found."
        )
    return temperature


def plot_mvsh(
    mvsh: MvsH | list[MvsH],
    normalized: bool = False,
    segment: Literal["", "virgin", "forward", "reverse", "loop"] = "",
    colors: str | list[str] = "auto",
    labels: str | list[str] | None = "auto",
    title: str = "",
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """Plots either a single M vs. H experiment or several on the same axes.

    Parameters
    ----------
    mvsh : MvsH | list[MvsH]
        The data to plot given as a single or list of `MvsH` objects.
    normalized : bool, optional
        If `True`, the magnetization will be normalized to the maximum value, by
        default False.
    segment : {"", "virgin", "forward", "reverse", "loop"}, optional
        If a segment is given, only that segment will be plotted, by default "".
    colors : str | list[str], optional
        A list of colors corresponding to the `MvsH` objects in `mvsh`, by default
        "auto". If "auto" and `mvsh` is a single `MvsH` object, the color will be
        black. If "auto" and `mvsh` is a list of `MvsH` objects with different
        temperatures, the colors will be a linear gradient from blue to red. If
        "auto" and `mvsh` is a list of `MvsH` objects with the same temperature, the
        colors will be the default `matplotlib` colors.
    labels : str | list[str] | None, optional
        The labels to assign the `MvsH` objects in the axes legend, by default "auto".
        If "auto", the labels will be the `temperature` of the `MvsH` objects.
    title : str, optional
        The title of the plot, by default "".
    **kwargs
        Keyword arguments mostly meant to affect the plot style. See
        `magnetopy.experiments.plot_utils.handle_options` for details.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
    """
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
            segment=segment,
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
        segment=segment,
        colors=colors,
        labels=labels,
        title=title,
        **kwargs,
    )


def plot_single_mvsh(
    mvsh: MvsH,
    normalized: bool = False,
    segment: str = "",
    color: str = "black",
    label: str | None = "auto",
    title: str = "",
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """Plots a single M vs. H experiment.

    Parameters
    ----------
    mvsh : MvsH
        The data to plot given as a single `MvsH` object.
    normalized : bool, optional
        If `True`, the magnetization will be normalized to the maximum value, by
        default False.
    segment : {"", "virgin", "forward", "reverse", "loop"}, optional
        If a segment is given, only that segment will be plotted, by default "".
    color : str | list[str], optional
        The color of the plot, by default "auto". If "auto", the color will be black.
    label : str | list[str] | None, optional
        The labels to assign the `MvsH` object in the axes legend, by default "auto".
        If "auto", the label will be the `temperature` of the `MvsH` object.
    title : str, optional
        The title of the plot, by default "".
    **kwargs
        Keyword arguments mostly meant to affect the plot style. See
        `magnetopy.experiments.plot_utils.handle_options` for details.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
    """
    options = handle_kwargs(**kwargs)

    color = "black" if color == "auto" else color

    fig, ax = plt.subplots()
    x = mvsh.simplified_data(segment)["field"] / 10000
    y = mvsh.simplified_data(segment)["moment"]
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

    handle_options(ax, options, label, title)

    force_aspect(ax)
    if options["save"]:
        plt.savefig(
            options["save"], dpi=300, bbox_inches="tight", facecolor="w", edgecolor="w"
        )
    return fig, ax


def plot_multiple_mvsh(
    mvsh: list[MvsH],
    normalized: bool = False,
    segment: str = "",
    colors: list[str] | Literal["auto"] = "auto",
    labels: list[str] | None = None,
    title: str = "",
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """Plots several M vs. H experiment on the same axes.

    Parameters
    ----------
    mvsh : MvsH | list[MvsH]
        The data to plot given as a list of `MvsH` objects.
    normalized : bool, optional
        If `True`, the magnetization will be normalized to the maximum value, by
        default False.
    segment : {"", "virgin", "forward", "reverse", "loop"}, optional
        If a segment is given, only that segment will be plotted, by default "".
    colors : str | list[str], optional
        A list of colors corresponding to the `MvsH` objects in `mvsh`, by default
        "auto". If "auto" and `mvsh` is a list of `MvsH` objects with different
        temperatures, the colors will be a linear gradient from blue to red. If
        "auto" and `mvsh` is a list of `MvsH` objects with the same temperature, the
        colors will be the default `matplotlib` colors.
    labels : str | list[str] | None, optional
        The labels to assign the `MvsH` objects in the axes legend, by default "auto".
        If "auto", the labels will be the `temperature` of the `MvsH` objects.
    title : str, optional
        The title of the plot, by default "".
    **kwargs
        Keyword arguments mostly meant to affect the plot style. See
        `magnetopy.experiments.plot_utils.handle_options` for details.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
    """
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
        x = m.simplified_data(segment)["field"] / 10000
        y = m.simplified_data(segment)["moment"]
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

    handle_options(ax, options, labels[0], title)
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
