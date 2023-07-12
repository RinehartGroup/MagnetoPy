from __future__ import annotations
from typing import Protocol
import pandas as pd

from magnetopy.data_files import DatFile
from magnetopy.parsing_utils import (
    label_clusters,
    unique_values,
    find_sequence_starts,
    find_temp_turnaround_point,
)


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
        file_data: pd.DataFrame,
        temperature: int | float,
        eps: float = 0.001,
        min_samples: int = 10,
        ndigits: int = 0,
    ) -> None:
        self.temperature = temperature
        self.data = self._set_data(file_data, eps, min_samples, ndigits)
        self.field_correction_file = ""

    def __str__(self) -> str:
        return f"MvsH at {self.temperature} K"

    def __repr__(self) -> str:
        return f"MvsH at {self.temperature} K"

    def _set_data(
        self, dat_file: DatFile, eps: float, min_samples: int, ndigits: int
    ) -> pd.DataFrame:
        file_data = dat_file.data
        file_data["cluster"] = label_clusters(
            file_data["Temperature (K)"], eps, min_samples
        )
        temps = unique_values(file_data["Temperature (K)"], eps, min_samples, ndigits)
        if self.temperature not in temps:
            raise self.TemperatureNotInDataError(
                f"Temperature {self.temperature} not in list of temperatures {temps}"
            )
        temperature_index = temps.index(self.temperature)
        cluster = file_data["cluster"].unique()[temperature_index]
        df = (
            file_data[file_data["cluster"] == cluster]
            .drop(columns=["cluster"])
            .reset_index(drop=True)
        )
        # set "uncorrected_moment" to be the moment directly from the dat file
        # whether the measurement was dc or vsm
        df["uncorrected_moment"] = df["Moment (emu)"].fillna(
            df["DC Moment Free Ctr (emu)"]
        )
        df["uncorrected_moment_err"] = df["M. Std. Err. (emu)"].fillna(
            df["DC Moment Err Free Ctr (emu)"]
        )
        return df

    def simplified_data(self, sequence: str = "") -> pd.DataFrame:
        # returns a dataframe with only the columns
        # time, temperature, field, moment, moment_err
        # sequence is one of: "", "virgin", "forward", "reverse", or "loop"
        pass

    def correct_field(self, field_correction_file: str) -> None:
        self.field_correction_file = field_correction_file
        # add "true_field" to the dataframe

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
        sequence_starts = find_sequence_starts(self.data["Magnetic Field (Oe)"])
        df = self.data.copy()
        segment = None
        if len(sequence_starts) == 3:
            # assume virgin -> reverse -> forward
            if sequence == "virgin":
                segment = df[sequence_starts[0] : sequence_starts[1]].reset_index(
                    drop=True
                )
            elif sequence == "reverse":
                segment = df[sequence_starts[1] : sequence_starts[2]].reset_index(
                    drop=True
                )
            elif sequence == "forward":
                segment = df[sequence_starts[2] :].reset_index(drop=True)
            elif sequence == "loop":
                segment = df[sequence_starts[1] :].reset_index(drop=True)
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
                    segment = df[sequence_starts[1] :].reset_index(drop=True)
            else:
                if sequence == "forward":
                    segment = df[sequence_starts[0] : sequence_starts[1]].reset_index(
                        drop=True
                    )
                elif sequence == "reverse":
                    segment = df[sequence_starts[1] :].reset_index(drop=True)
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


class ZFCFC:
    def __init__(self, dat_file: DatFile, turnaround: int, experiment: str) -> None:
        self.data = self._set_data(dat_file.data.copy(), turnaround, experiment)
        self.field = dat_file.data["Magnetic Field (Oe)"].mean()

    def __str__(self) -> str:
        return f"{self.__class__.__name__} at {self.field} Oe"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} at {self.field} Oe"

    def _set_data(
        self, data: pd.DataFrame, turnaround: int, experiment: str
    ) -> pd.DataFrame:
        if experiment == "zfc":
            df = data.iloc[:turnaround].reset_index(drop=True)
        else:
            df = data.iloc[turnaround:].reset_index(drop=True)
        df["uncorrected_moment"] = df["Moment (emu)"].fillna(
            df["DC Moment Free Ctr (emu)"]
        )
        df["uncorrected_moment_err"] = df["M. Std. Err. (emu)"].fillna(
            df["DC Moment Err Free Ctr (emu)"]
        )
        return df

    def simplified_data(self) -> pd.DataFrame:
        pass


class ZFC(ZFCFC):
    def __init__(self, dat_file: DatFile, data_contents: str = "zfcfc") -> None:
        # if data_contents is not "zfcfc" then the data is only the zfc data
        # turnaround is the end of the zfc data
        turnaround = (
            find_temp_turnaround_point(dat_file.data)
            if data_contents == "zfcfc"
            else len(dat_file.data)
        )
        super().__init__(dat_file, turnaround, "zfc")


class FC(ZFCFC):
    def __init__(self, dat_file: DatFile, data_contents: str = "zfcfc") -> None:
        # if data_contents is not "zfcfc" then the data is only the fc data
        # turnaround is the beginning of the fc data
        turnaround = (
            find_temp_turnaround_point(dat_file.data) if data_contents == "zfcfc" else 0
        )
        super().__init__(dat_file, turnaround, "fc")
