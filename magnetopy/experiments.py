from __future__ import annotations
import json
from pathlib import Path
import re
from typing import Protocol
import warnings
import numpy as np
import pandas as pd

from magnetopy.data_files import DatFile
from magnetopy.parsing_utils import (
    label_clusters,
    unique_values,
    find_sequence_starts,
    find_temp_turnaround_point,
)


class TemperatureDetectionError(Exception):
    pass


class FieldDetectionError(Exception):
    pass


class FileNameWarning(UserWarning):
    pass


class Experiment(Protocol):
    data: pd.DataFrame

    def simplified_data(self, *args, **kwargs) -> pd.DataFrame:
        ...


class MvsH:
    class TemperatureNotInDataError(Exception):
        pass

    class SegmentError(Exception):
        pass

    def __init__(
        self,
        dat_file: str | Path | DatFile,
        temperature: int | float | None = None,
        **kwargs,
    ) -> None:
        if not isinstance(dat_file, DatFile):
            dat_file = DatFile(Path(dat_file))
        self.origin_file = dat_file.local_path.name

        # optional arguments used for algorithmic separation of
        # data at the requested temperature
        n_digits = _num_digits_after_decimal(temperature) if temperature else 0
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
        _add_uncorrected_moment_columns(self)
        self.field_correction_file = ""
        self.scaling = []
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
        _scale_dc_data(
            self,
            mass,
            eicosane_mass,
            molecular_weight,
            diamagnetic_correction,
        )

    def correct_field(self, field_correction_file: str | Path) -> None:
        pd_mvsh = TrueFieldCorrection(field_correction_file)
        self.field_correction_file = pd_mvsh.origin_file
        self.data["true_field"] = pd_mvsh.data["true_field"]

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

    @classmethod
    def get_all_in_file(
        cls,
        dat_file: str | Path | DatFile,
        eps: float = 0.001,
        min_samples: int = 10,
        ndigits: int = 0,
    ) -> list[MvsH]:
        if not isinstance(dat_file, DatFile):
            dat_file = DatFile(Path(dat_file))
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


class ZFCFC:
    class NonMatchingFieldError(Exception):
        pass

    def __init__(
        self,
        dat_file: str | Path | DatFile,
        experiment: str,
        field: int | float | None = None,
        **kwargs,
    ) -> None:
        if not isinstance(dat_file, DatFile):
            dat_file = DatFile(Path(dat_file))

        n_digits = _num_digits_after_decimal(field) if field else 0
        options = {"n_digits": n_digits, "suppress_warnings": False}
        options.update(kwargs)

        filename_label = _filename_label(
            dat_file.local_path.name, experiment, options["suppress_warnings"]
        )

        if field is None:
            field = _auto_detect_field(dat_file, experiment, options["n_digits"])
        self.field = field

        if dat_file.comments:
            self.data = self._set_data_from_comments(dat_file, experiment)
        else:
            if filename_label in ["zfcfc", "unknown"]:
                self.data = self._set_data_auto(dat_file, experiment)
            else:
                self.data = self._set_single_sequence_data(
                    dat_file, experiment, options["n_digits"]
                )
        _add_uncorrected_moment_columns(self)
        self.scaling = []

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
        _scale_dc_data(
            self,
            mass,
            eicosane_mass,
            molecular_weight,
            diamagnetic_correction,
        )

    def simplified_data(self) -> pd.DataFrame:
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

    @classmethod
    def get_all_zfcfc_in_file(
        cls,
        dat_file: str | Path | DatFile,
        experiment: str,
        n_digits: int = 0,
    ) -> list[ZFCFC]:
        if not isinstance(dat_file, DatFile):
            dat_file = DatFile(Path(dat_file))
        if dat_file.comments:
            zfcfc_objs = cls._get_all_zfcfc_in_commented_file(dat_file, experiment)

        else:
            zfcfc_objs = cls._get_all_zfcfc_in_uncommented_file(
                dat_file,
                experiment,
                n_digits,
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
        return [ZFCFC(dat_file, experiment, n_digits=n_digits)]


class ZFC(ZFCFC):
    def __init__(
        self, dat_file: str | Path | DatFile, field: int | float | None = None, **kwargs
    ) -> None:
        super().__init__(dat_file, "zfc", field, **kwargs)

    @classmethod
    def get_all_in_file(
        cls,
        dat_file: str | Path | DatFile,
        n_digits: int = 0,
    ) -> list[ZFC]:
        return ZFCFC.get_all_zfcfc_in_file(dat_file, "zfc", n_digits)


class FC(ZFCFC):
    def __init__(
        self, dat_file: str | Path | DatFile, field: int | float | None = None, **kwargs
    ) -> None:
        super().__init__(dat_file, "fc", field, **kwargs)

    @classmethod
    def get_all_in_file(
        cls,
        dat_file: str | Path | DatFile,
        n_digits: int = 0,
    ) -> list[FC]:
        return ZFCFC.get_all_zfcfc_in_file(dat_file, "fc", n_digits)


def _num_digits_after_decimal(number: int | float):
    if isinstance(number, int):
        return 0
    return len(str(number).split(".")[1])


def _filename_label(filename: str, experiment: str, suppress_warnings: bool) -> str:
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
    if label in ["zfc", "fc"] and label != experiment and not suppress_warnings:
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


class _DcExperiment(Protocol):
    data: pd.DataFrame
    scaling: list[str]


def _add_uncorrected_moment_columns(experiment: _DcExperiment) -> None:
    # set "uncorrected_moment" to be the moment directly from the dat file
    # whether the measurement was dc or vsm
    experiment.data["uncorrected_moment"] = experiment.data["Moment (emu)"].fillna(
        experiment.data["DC Moment Free Ctr (emu)"]
    )
    experiment.data["uncorrected_moment_err"] = experiment.data[
        "M. Std. Err. (emu)"
    ].fillna(experiment.data["DC Moment Err Free Ctr (emu)"])


def _scale_dc_data(
    experiment: _DcExperiment,
    mass: float = 0,
    eicosane_mass: float = 0,
    molecular_weight: float = 0,
    diamagnetic_correction: float = 0,
) -> None:
    if mass and molecular_weight:
        experiment.scaling.append("molar")
        if eicosane_mass:
            experiment.scaling.append("eicosane")
        if diamagnetic_correction:
            experiment.scaling.append("diamagnetic_correction")
        mol = mass / molecular_weight
        _scale_magnetic_data_molar_w_eicosane_and_diamagnet(
            experiment.data, mol, eicosane_mass, diamagnetic_correction
        )
    elif mass:
        experiment.scaling.append("mass")
        _scale_magnetic_data_mass(experiment.data, mass)


def _scale_magnetic_data_molar_w_eicosane_and_diamagnet(
    data: pd.DataFrame,
    mol_sample: float,
    eicosane_mass: float,
    diamagnetic_correction: float,
) -> None:
    mol_eicosane = eicosane_mass / 282.55 if eicosane_mass else 0
    eicosane_diamagnetism = (
        -0.00024306 * mol_eicosane
    )  # eicosane chi_D = -0.00024306 emu/mol
    sample_molar_diamagnetism = (
        mol_sample * diamagnetic_correction if diamagnetic_correction else 0
    )
    # chi in units of cm^3/mol
    data["chi"] = (
        data["uncorrected_moment"] / data["Magnetic Field (Oe)"] - eicosane_diamagnetism
    ) / mol_sample - sample_molar_diamagnetism
    data["chi_err"] = (
        data["uncorrected_moment_err"] / data["Magnetic Field (Oe)"]
        - eicosane_diamagnetism
    ) / mol_sample - sample_molar_diamagnetism
    # chiT in units of cm3 K mol-1
    data["chi_t"] = data["chi"] * data["Temperature (K)"]
    data["chi_t_err"] = data["chi_err"] * data["Temperature (K)"]
    # moment in units of Bohr magnetons
    data["moment"] = data["chi"] * data["Magnetic Field (Oe)"] / 5585
    data["moment_err"] = data["chi_err"] * data["Magnetic Field (Oe)"] / 5585


def _scale_magnetic_data_mass(data: pd.DataFrame, mass: float) -> None:
    # moment in units of emu/g
    data["moment"] = data["uncorrected_moment"] / (mass / 1000)
    data["moment_err"] = data["uncorrected_moment_err"] / (mass / 1000)
    # chi in units of cm^3/g
    data["chi"] = data["moment"] / data["Magnetic Field (Oe)"]
    data["chi_err"] = data["moment_err"] / data["Magnetic Field (Oe)"]
    # chiT in units of cm3 K g-1
    data["chi_t"] = data["chi"] * data["Temperature (K)"]
    data["chi_t_err"] = data["chi_err"] * data["Temperature (K)"]
