import csv
import hashlib
from pathlib import Path
from typing import Any
from collections import OrderedDict
from datetime import datetime

import pandas as pd


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

    def __init__(self, file_path: str | Path) -> None:
        super().__init__(file_path, "magnetometry")
        self.header = self._read_header()
        self.data = self._read_data()
        self.comments = self._get_comments()
        self.date_created = self._get_date_created()
        self.experiments_in_file = self._get_experiments_in_file()

    def __str__(self) -> str:
        return f"DatFile({self.local_path.name})"

    def __repr__(self) -> str:
        return f"DatFile({self.local_path.name})"

    def _read_header(self, delimiter: str = "\t") -> list[list[str]]:
        header: list[list[str]] = []
        with self.local_path.open() as f:
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
        else:
            if len(self.data["Magnetic Field (Oe)"].unique()) == 1:
                experiments.append("zfcfc")
            else:
                experiments.append("mvsh")
        return experiments

    def as_dict(self) -> dict[str, Any]:
        """Serializes the DatFile object to a dictionary.

        Returns
        -------
        dict[str, Any]
            Contains the following keys: local_path, length, date_created, sha512,
            experiments_in_file.
        """
        return {
            "experiment_type": self.experiment_type,
            "local_path": str(self.local_path),
            "length": self.length,
            "date_created": self.date_created.isoformat(),
            "sha512": self.sha512,
            "experiments_in_file": self.experiments_in_file,
        }
