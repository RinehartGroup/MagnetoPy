from __future__ import annotations
from pathlib import Path
import re
from typing import Any, Literal
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

from magnetopy.data_files import DatFile, filename_label, plot_raw, plot_raw_residual
from magnetopy.experiments.plot_utils import get_ylabel, handle_kwargs, handle_options
from magnetopy.experiments.utils import (
    add_uncorrected_moment_columns,
    num_digits_after_decimal,
    scale_dc_data,
)
from magnetopy.parsing_utils import find_temp_turnaround_point
from magnetopy.plot_utils import default_colors, force_aspect, linear_color_gradient


class FieldDetectionError(Exception):
    pass


class ZFCFC:
    """A single ZFCFC experiment at a given field. Unlikely to be called directly,
    instead use `ZFC` or `FC`.

    Parameters
    ----------
    dat_file : str, Path, or DatFile
        The .dat file containing the data for the experiment.
    experiment : {"zfc", "fc"}
        The type of experiment. Either "zfc" or "fc".
    field : int or float, optional
        The magnetic field of the experiment in Oe. Requied if the .dat file contains
        multiple uncommented experiments at different fields. If `None` and the .dat
        file contains a single experiment, the field will be automatically detected.
        Defaults to `None`.
    parse_raw : bool, optional
        If `True` and there is a corresponding .rw.dat file, the raw data will be
        parsed and added to the `data` attribute. Defaults to `False`.
    **kwargs : dict, optional
        Keyword arguments used for algorithmic separation of data at the requested
        field.
        - n_digits : int, optional
        - suppress_warnings : bool, optional

    Attributes
    ----------
    origin_file : str
        The name of the .dat file from which the data was parsed.
    field : float
        The field of the experiment in Oe.
    data : pandas.DataFrame
        The data from the experiment. Columns are taken directly from the .dat file.
    scaling : list of str
        The scaling applied to the data. If no scaling has been applied, this will be
        an empty list. Possible values are: `"mass"`, `"molar"`, `"eicosane"`,
        and `"diamagnetic_correction"`.
    temperature_range : tuple of float
        The minimum and maximum temperature values in the data.

    Raises
    ------
    NonMatchingFieldError
        Raised if the requested field is not in the data.
    """

    class NonMatchingFieldError(Exception):
        pass

    def __init__(
        self,
        dat_file: str | Path | DatFile,
        experiment: Literal["zfc", "fc"],
        field: int | float | None = None,
        parse_raw: bool = False,
        **kwargs,
    ) -> None:
        if not isinstance(dat_file, DatFile):
            dat_file = DatFile(Path(dat_file), parse_raw)
        self.origin_file = dat_file.local_path.name

        n_digits = num_digits_after_decimal(field) if field else 0
        options = {"n_digits": n_digits, "suppress_warnings": False}
        options.update(kwargs)

        found_filename_label = filename_label(
            dat_file.local_path.name, experiment, options["suppress_warnings"]
        )

        if field is None:
            field = _auto_detect_field(dat_file, experiment, options["n_digits"])
        self.field = field

        if dat_file.comments:
            self.data = self._set_data_from_comments(dat_file, experiment)
        else:
            if found_filename_label in ["zfcfc", "unknown"]:
                self.data = self._set_data_auto(dat_file, experiment)
            else:
                self.data = self._set_single_sequence_data(
                    dat_file, experiment, options["n_digits"]
                )
        add_uncorrected_moment_columns(self)
        self.scaling = []
        self.temperature_range = self._determine_temperature_range()

    def __str__(self) -> str:
        return f"{self.__class__.__name__} at {self.field} Oe"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} at {self.field} Oe"

    def _set_data_from_comments(
        self, dat_file: DatFile, experiment: str
    ) -> pd.DataFrame:
        start_idx: int | None = None
        end_idx: int | None = None
        for comment_idx, (data_idx, comment_list) in enumerate(
            dat_file.comments.items()
        ):
            # ignore other experiments
            if experiment not in map(str.lower, comment_list):
                continue
            # one of the comments should be a number denoting the field
            # may also include a unit, e.g. "1000 Oe"
            for comment in comment_list:
                if match := re.search(r"\d+", comment):
                    found_field = float(match.group())
                    # check to see if the unit is T otherwise assume Oe
                    if "T" in comment:
                        found_field = found_field * 1e4
                    if found_field == self.field:
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
            raise self.NonMatchingFieldError(
                f"Temperature {self.field} not in data in {dat_file}. "
                "Or the comments are not formatted correctly."
            )
        df = dat_file.data.iloc[start_idx:end_idx].reset_index(drop=True)
        return df

    def _set_data_auto(self, dat_file: DatFile, experiment: str) -> pd.DataFrame:
        turnaround = find_temp_turnaround_point(dat_file.data)
        # assume zfc, then fc
        if experiment == "zfc":
            df = dat_file.data.iloc[:turnaround].reset_index(drop=True)
        else:
            df = dat_file.data.iloc[turnaround:].reset_index(drop=True)
        return df

    def _set_single_sequence_data(
        self, dat_file: DatFile, experiment: str, n_digits: int
    ) -> pd.DataFrame:
        """
        Used for when the file contains a single sequence of data, e.g. a single ZFC or FC
        at a single field."""
        df = dat_file.data.copy()
        found_fields = np.unique(df["Magnetic Field (Oe)"])
        if len(found_fields) != 1:
            raise ZFCFC.NonMatchingFieldError(
                f"Attempting to read in {experiment} data from {dat_file}, "
                f"but found data from multiple fields ({found_fields}). "
                "This method currently only supports files containing data from a single "
                "field."
            )
        if round(found_fields[0], n_digits) != self.field:
            raise ZFCFC.NonMatchingFieldError(
                f"Attempting to read in {experiment} data from {dat_file}, "
                f"but found data from a different field ({found_fields[0]}) "
                f"than the one specified ({self.field})."
            )
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

        See Also
        --------
        magnetopy.experiments.utils.scale_dc_data
        """
        scale_dc_data(
            self,
            mass,
            eicosane_mass,
            molecular_weight,
            diamagnetic_correction,
        )

    def simplified_data(self) -> pd.DataFrame:
        """Returns a simplified version of the data, removing unnecessary columns
        and renaming the remaining columns to more convenient names.

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
        full_df = self.data.copy()
        df = pd.DataFrame()
        df["time"] = full_df["Time Stamp (sec)"]
        df["temperature"] = full_df["Temperature (K)"]
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

    def _determine_temperature_range(self) -> tuple[float, float]:
        simplified_data = self.simplified_data()
        return (
            simplified_data["temperature"].min(),
            simplified_data["temperature"].max(),
        )

    def plot_raw(self, *args, **kwargs) -> tuple[plt.Figure, plt.Axes]:
        """Plots the raw voltage data from the experiment.

        Returns
        -------
        tuple[plt.Figure, plt.Axes]

        See Also
        --------
        magnetopy.data_files.plot_raw
        """
        return plot_raw(self.data, *args, **kwargs)

    def plot_raw_residual(self, *args, **kwargs) -> tuple[plt.Figure, plt.Axes]:
        """Plots the residual of the raw voltage data from the experiment.

        Returns
        -------
        tuple[plt.Figure, plt.Axes]

        See Also
        --------
        magnetopy.data_files.plot_raw_residual
        """
        return plot_raw_residual(self.data, *args, **kwargs)

    def as_dict(self) -> dict[str, Any]:
        """Returns a dictionary representation of the `MvsH` object.

        Returns
        -------
        dict[str, Any]
            Keys are: `"origin_file"`, `"field"`, `"temperature_range"`,
            and `"scaling"`.
        """
        return {
            "origin_file": self.origin_file,
            "field": self.field,
            "temperature_range": self.temperature_range,
            "scaling": self.scaling,
        }

    @classmethod
    def get_all_zfcfc_in_file(
        cls,
        dat_file: str | Path | DatFile,
        experiment: str,
        n_digits: int = 0,
        parse_raw: bool = False,
    ) -> list[ZFCFC]:
        """Given a .dat file that contains one or more ZFC or FC experiments, returns a
        list of `ZFC` or `FC` objects, one for each experiment.

        Parameters
        ----------
        dat_file : str | Path | DatFile
            The .dat file containing the data for the experiment.
        experiment : {"zfc", "fc"}
            The type of experiment. Either "zfc" or "fc".
        n_digits : int, optional
            The number of digits after the decimal point to round the field to. Defaults
            to `0`.
        parse_raw : bool, optional
            If `True` and there is a corresponding .rw.dat file, the raw data will be
            parsed and added to the `data` attribute. Defaults to `False`.

        Returns
        -------
        list[ZFC] | list[FC]
            A list of `ZFC` or `FC` objects, one for each experiment in the .dat file,
            sorted by increasing field.
        """
        if not isinstance(dat_file, DatFile):
            dat_file = DatFile(Path(dat_file), parse_raw)
        if dat_file.comments:
            zfcfc_objs = cls._get_all_zfcfc_in_commented_file(dat_file, experiment)

        else:
            zfcfc_objs = cls._get_all_zfcfc_in_uncommented_file(
                dat_file, experiment, n_digits
            )
        zfcfc_objs.sort(key=lambda x: x.field)
        return zfcfc_objs

    @classmethod
    def _get_all_zfcfc_in_commented_file(
        cls,
        dat_file: DatFile,
        experiment: str,
    ) -> list[ZFCFC]:
        zfcfc_objs = []
        for comment_list in dat_file.comments.values():
            # ignore other experiments
            if experiment not in map(str.lower, comment_list):
                continue
            # one of the comments should be a number denoting the field
            # may also include a unit, e.g. "1000 Oe"
            for comment in comment_list:
                if match := re.search(r"\d+", comment):
                    field = float(match.group())
                    # check to see if the unit is T otherwise assume Oe
                    if "T" in comment:
                        field = field * 1e4
                    # the following type guard allows for the `get_all` method to be
                    # called from the parent (ZFCFC) or child classes (ZFC, FC)
                    if cls.__base__ == ZFCFC:
                        # we're calling from ZFC or FC
                        zfcfc_objs.append(cls(dat_file, field))
                    else:
                        # we're calling from ZFCFC
                        zfcfc_objs.append(cls(dat_file, experiment, field))
        return zfcfc_objs

    @classmethod
    def _get_all_zfcfc_in_uncommented_file(
        cls,
        dat_file: DatFile,
        experiment: str,
        n_digits: int,
    ) -> list[ZFCFC]:
        """This method currently only supports an uncommented file with a single experiment"""
        # the following type guard allows for the `get_all` method to be
        # called from the parent (ZFCFC) or child classes (ZFC, FC)
        zfcfc_objs = []
        if cls.__base__ == ZFCFC:
            # we're calling from ZFC or FC
            zfcfc_objs.append(
                cls(  # pylint: disable=E1120
                    dat_file,
                    n_digits=n_digits,
                )
            )
        else:
            # we're calling from ZFCFC
            zfcfc_objs.append(cls(dat_file, experiment, n_digits=n_digits))
        return zfcfc_objs


class ZFC(ZFCFC):
    """A single ZFC experiment at a given field.

    Parameters
    ----------
    dat_file : str, Path, or DatFile
        The .dat file containing the data for the experiment.
    field : int or float, optional
        The magnetic field of the experiment in Oe. Requied if the .dat file contains
        multiple uncommented experiments at different fields. If `None` and the .dat
        file contains a single experiment, the field will be automatically detected.
        Defaults to `None`.
    parse_raw : bool, optional
        If `True` and there is a corresponding .rw.dat file, the raw data will be
        parsed and added to the `data` attribute. Defaults to `False`.
    **kwargs : dict, optional
        Keyword arguments used for algorithmic separation of data at the requested
        field.
        - n_digits : int, optional
        - suppress_warnings : bool, optional

    Attributes
    ----------
    origin_file : str
        The name of the .dat file from which the data was parsed.
    field : float
        The field of the experiment in Oe.
    data : pandas.DataFrame
        The data from the experiment. Columns are taken directly from the .dat file.
    scaling : list of str
        The scaling applied to the data. If no scaling has been applied, this will be
        an empty list. Possible values are: `"mass"`, `"molar"`, `"eicosane"`,
        and `"diamagnetic_correction"`.
    temperature_range : tuple of float
        The minimum and maximum temperature values in the data.

    Raises
    ------
    NonMatchingFieldError
        Raised if the requested field is not in the data.
    """

    def __init__(
        self,
        dat_file: str | Path | DatFile,
        field: int | float | None = None,
        parse_raw: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(dat_file, "zfc", field, parse_raw, **kwargs)

    @classmethod
    def get_all_in_file(
        cls,
        dat_file: str | Path | DatFile,
        n_digits: int = 0,
        parse_raw: bool = False,
    ) -> list[ZFC]:
        return super().get_all_zfcfc_in_file(dat_file, "zfc", n_digits, parse_raw)

    def __str__(self):
        return f"ZFC at {self.field} Oe"

    def __repr__(self):
        return f"ZFC at {self.field} Oe"


class FC(ZFCFC):
    """A single FC experiment at a given field.

    Parameters
    ----------
    dat_file : str, Path, or DatFile
        The .dat file containing the data for the experiment.
    field : int or float, optional
        The magnetic field of the experiment in Oe. Requied if the .dat file contains
        multiple uncommented experiments at different fields. If `None` and the .dat
        file contains a single experiment, the field will be automatically detected.
        Defaults to `None`.
    parse_raw : bool, optional
        If `True` and there is a corresponding .rw.dat file, the raw data will be
        parsed and added to the `data` attribute. Defaults to `False`.
    **kwargs : dict, optional
        Keyword arguments used for algorithmic separation of data at the requested
        field.
        - n_digits : int, optional
        - suppress_warnings : bool, optional

    Attributes
    ----------
    origin_file : str
        The name of the .dat file from which the data was parsed.
    field : float
        The field of the experiment in Oe.
    data : pandas.DataFrame
        The data from the experiment. Columns are taken directly from the .dat file.
    scaling : list of str
        The scaling applied to the data. If no scaling has been applied, this will be
        an empty list. Possible values are: `"mass"`, `"molar"`, `"eicosane"`,
        and `"diamagnetic_correction"`.
    temperature_range : tuple of float
        The minimum and maximum temperature values in the data.

    Raises
    ------
    NonMatchingFieldError
        Raised if the requested field is not in the data.
    """

    def __init__(
        self,
        dat_file: str | Path | DatFile,
        field: int | float | None = None,
        parse_raw: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(dat_file, "fc", field, parse_raw, **kwargs)

    @classmethod
    def get_all_in_file(
        cls,
        dat_file: str | Path | DatFile,
        n_digits: int = 0,
        parse_raw: bool = False,
    ) -> list[FC]:
        return super().get_all_zfcfc_in_file(dat_file, "fc", n_digits, parse_raw)

    def __str__(self):
        return f"FC at {self.field} Oe"

    def __repr__(self):
        return f"FC at {self.field} Oe"


def _auto_detect_field(
    dat_file: DatFile, experiment: str, n_digits: int
) -> int | float:
    field: float | None = None
    if dat_file.comments:
        exp_comments = []
        for comment_list in dat_file.comments.values():
            if experiment in map(str.lower, comment_list):
                exp_comments.append(comment_list)
        if len(exp_comments) != 1:
            raise FieldDetectionError(
                f"Could not autodetect field for {experiment}. When not specificying "
                "a field, the DatFile must contain exactly one field. Found "
                f"{len(exp_comments)} fields."
            )
        comments = exp_comments[0]
        for comment in comments:
            if match := re.search(r"\d+", comment):
                found_field = float(match.group())
                # check to see if the unit is T otherwise assume Oe
                if "T" in comment:
                    found_field = found_field * 1e4
                field = found_field
    else:
        fields = np.unique(dat_file.data["Magnetic Field (Oe)"])
        if len(fields) != 1:
            raise FieldDetectionError(
                f"Could not autodetect field for {experiment}. When not specificying "
                "a field, the DatFile must contain exactly one field. Found "
                f"{len(fields)} fields."
            )
        field = round(fields[0], n_digits)
    if field is None:
        raise FieldDetectionError(
            f"Could not autodetect field for {experiment}. Please specify a field."
        )
    if n_digits == 0:
        field = int(field)
    return field


def plot_zfcfc(
    zfc: ZFC | list[ZFC],
    fc: FC | list[FC],
    y_val: Literal["moment", "chi", "chi_t"] = "moment",
    normalized: bool = False,
    colors: str | list[str] = "auto",
    labels: str | list[str] | None = "auto",
    title: str = "",
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """Plots either a single `ZFC`/`FC` pair or several on the same axes.

    Parameters
    ----------
    zfc : ZFC | list[ZFC]
        The ZFC data to plot. Must be the same length as `fc`.
    fc : FC | list[FC]
        The FC data to plot. Must be the same length as `zfc`.
    normalized : bool, optional
        If `True`, the magnetization will be normalized to the maximum value, by
        default False.
    colors : str | list[str], optional
        A list of colors corresponding to the `ZFC`/`FC` pairs, by default "auto". If
        "auto" and there is a single pair, the color will be black. If "auto" and
        there is a list of pairs with different fields, the colors will be a linear
        gradient from purple to green. If "auto" and the list of pairs is at the same
        field, the colors will be the default `matplotlib` colors.
    labels : str | list[str] | None, optional
        The labels to assign the `ZFC`/`FC` pairs in the axes legend, by default "auto".
        If "auto", the labels will be the `field` of the `ZFC`/`FC` objects.
    title : str, optional
        The title of the plot, by default "".
    **kwargs
        Keyword arguments mostly meant to affect the plot style. See `handle_options`
        for details.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]

    See Also
    --------
    magnetopy.experiments.plot_utils.handle_options
    """
    if isinstance(zfc, list) and len(zfc) == 1:
        zfc = zfc[0]
    if isinstance(fc, list) and len(fc) == 1:
        fc = fc[0]
    if isinstance(zfc, ZFC) and isinstance(fc, FC):
        if isinstance(colors, list) or isinstance(labels, list):
            raise ValueError(
                "If plotting a single ZFCFC, `colors` and `labels` must be a single value"
            )
        return plot_single_zfcfc(
            zfc=zfc,
            fc=fc,
            y_val=y_val,
            normalized=normalized,
            color="black" if colors == "auto" else colors,
            label=labels,
            title=title,
            **kwargs,
        )
    if not isinstance(zfc, list) or not isinstance(fc, list) or (len(zfc) != len(fc)):
        raise ValueError("ZFC and FC must be the same length")
    if colors != "auto" and not isinstance(colors, list):
        raise ValueError(
            "If plotting multiple ZFCFC, `colors` must be a list or 'auto'."
        )
    if labels is not None and labels != "auto" and not isinstance(labels, list):
        raise ValueError(
            "If plotting multiple ZFCFC, `labels` must be a list or 'auto' or `None`."
        )
    zfc.sort(key=lambda x: x.field)
    fc.sort(key=lambda x: x.field)
    for zfc_i, fc_i in zip(zfc, fc):
        if zfc_i.field != fc_i.field:
            raise ValueError("ZFC and FC must have the same fields")
    return plot_multiple_zfcfc(
        zfc,
        fc,
        y_val=y_val,
        normalized=normalized,
        colors=colors,
        labels=labels,
        title=title,
        **kwargs,
    )


def plot_single_zfcfc(
    zfc: ZFC,
    fc: FC,
    y_val: Literal["moment", "chi", "chi_t"] = "moment",
    normalized: bool = False,
    color: str = "black",
    label: str | None = "auto",
    title: str = "",
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """Plots a single `ZFC`/`FC` pair.

    Parameters
    ----------
    zfc : ZFC
        The ZFC data to plot.
    fc : FC
        The FC data to plot.
    normalized : bool, optional
        If `True`, the magnetization will be normalized to the maximum value, by
        default False.
    colors : str | list[str], optional
        A list of colors corresponding to the `ZFC`/`FC` pairs, by default "auto". If
        "auto" and there is a single pair, the color will be black. If "auto" and
        there is a list of pairs with different fields, the colors will be a linear
        gradient from purple to green. If "auto" and the list of pairs is at the same
        field, the colors will be the default `matplotlib` colors.
    labels : str | list[str] | None, optional
        The labels to assign the `ZFC/`FC` pair in the axes legend, by default "auto".
        If "auto", the label will be the `field` of the `MvsH` object.
    title : str, optional
        The title of the plot, by default "".
    **kwargs
        Keyword arguments mostly meant to affect the plot style. See `handle_options`
        for details.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]

    See Also
    --------
    magnetopy.experiments.plot_utils.handle_options
    """
    options = handle_kwargs(**kwargs)

    fig, ax = plt.subplots()
    x_zfc = zfc.simplified_data()["temperature"]
    y_zfc = zfc.simplified_data()[y_val]
    y_zfc = y_zfc / y_zfc.max() if normalized else y_zfc
    x_fc = fc.simplified_data()["temperature"]
    y_fc = fc.simplified_data()[y_val]
    y_fc = y_fc / y_fc.max() if normalized else y_fc
    if label is None:
        ax.plot(x_zfc, y_zfc, c=color, ls="--")
        ax.plot(x_fc, y_fc, c=color)
    else:
        if label == "auto":
            if zfc.field > 10000:
                label = f"{(zfc.field / 10000):.0f} T"
            else:
                label = f"{zfc.field:.0f} Oe"
        ax.plot(x_zfc, y_zfc, c=color, ls="--", label=label)
        ax.plot(x_fc, y_fc, c=color, label="")

    ax.set_xlabel("Temperature (K)")
    if normalized:
        if y_val == "moment":
            normalized_ylabel = "Normalized Magnetization"
        elif y_val == "chi":
            normalized_ylabel = r"Normalized $\chi$"
        else:
            normalized_ylabel = r"Normalized $\chi\cdot$T"
        ax.set_ylabel(normalized_ylabel)
    else:
        ylabel = get_ylabel(y_val, zfc.scaling)
        ax.set_ylabel(ylabel)

    handle_options(ax, label, title, options)

    force_aspect(ax)
    if options["save"]:
        plt.savefig(
            options["save"], dpi=300, bbox_inches="tight", facecolor="w", edgecolor="w"
        )
    return fig, ax


def plot_multiple_zfcfc(
    zfc: list[ZFC],
    fc: list[FC],
    y_val: Literal["moment", "chi", "chi_t"] = "moment",
    normalized: bool = False,
    colors: list[str] | Literal["auto"] = "auto",
    labels: list[str] | None = None,
    title: str = "",
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """Plots several `ZFC`/`FC` pairs on the same axes.

    Parameters
    ----------
    zfc : list[ZFC]
        The ZFC data to plot. Must be the same length as `fc`.
    fc : list[FC]
        The FC data to plot. Must be the same length as `zfc`.
    normalized : bool, optional
        If `True`, the magnetization will be normalized to the maximum value, by
        default False.
    colors : str | list[str], optional
        A list of colors corresponding to the `ZFC`/`FC` pairs, by default "auto".
        If "auto" and there is a list of pairs with different fields, the colors will
        be a linear gradient from purple to green. If "auto" and the list of pairs is
        at the same field, the colors will be the default `matplotlib` colors.
    labels : str | list[str] | None, optional
        The labels to assign the `ZFC`/`FC` pairs in the axes legend, by default "auto".
        If "auto", the labels will be the `field` of the `ZFC`/`FC` objects.
    title : str, optional
        The title of the plot, by default "".
    **kwargs
        Keyword arguments mostly meant to affect the plot style. See `handle_options`
        for details.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]

    See Also
    --------
    magnetopy.experiments.plot_utils.handle_options
    """
    options = handle_kwargs(**kwargs)

    if colors == "auto":
        colors = default_colors(len(zfc))
    if _check_if_variable_field(zfc):
        zfc.sort(key=lambda x: x.field)
        colors = linear_color_gradient("purple", "green", len(zfc))
        if labels == "auto":
            labels = [f"{x.field:.0f} Oe" for x in zfc]
    if labels is None:
        labels: list[None] = [None] * len(zfc)

    fig, ax = plt.subplots()
    for zfc_i, fc_i, color, label in zip(zfc, fc, colors, labels):
        x_zfc = zfc_i.simplified_data()["temperature"]
        y_zfc = zfc_i.simplified_data()[y_val]
        y_zfc = y_zfc / y_zfc.max() if normalized else y_zfc
        x_fc = fc_i.simplified_data()["temperature"]
        y_fc = fc_i.simplified_data()[y_val]
        y_fc = y_fc / y_fc.max() if normalized else y_fc
        if label:
            ax.plot(x_zfc, y_zfc, c=color, ls="--", label=label)
            ax.plot(x_fc, y_fc, c=color, label="")
        else:
            ax.plot(x_zfc, y_zfc, c=color, ls="--")
            ax.plot(x_fc, y_fc, c=color)

    ax.set_xlabel("Temperature (K)")
    if normalized:
        if y_val == "moment":
            normalized_ylabel = "Normalized Magnetization"
        elif y_val == "chi":
            normalized_ylabel = r"Normalized $\chi$"
        else:
            normalized_ylabel = r"Normalized $\chi\cdot$T"
        ax.set_ylabel(normalized_ylabel)
    else:
        ylabel = get_ylabel(y_val, zfc[0].scaling)
        ax.set_ylabel(ylabel)

    handle_options(ax, labels[0], title, options)

    force_aspect(ax)
    if options["save"]:
        plt.savefig(
            options["save"], dpi=300, bbox_inches="tight", facecolor="w", edgecolor="w"
        )
    return fig, ax


def _check_if_variable_field(zfc: list[ZFC]):
    first_field = zfc[0].field
    for zfc_obj in zfc:
        if zfc_obj.field != first_field:
            return True
    return False
