from typing import Protocol

import pandas as pd


class DcExperiment(Protocol):
    """The protocol for a DC magnetometry experiment, e.g. M vs H, ZFC/FC, etc.

    Attributes
    ----------
    data : pd.DataFrame
        The data as read in from the experiment file. It should include the columns
        `"uncorrected_moment"` and `"uncorrected_moment_err"`, which are the moment
        and moment error directly from the experiment file, whether the measurement
        was DC or VSM. Scaling methods will act on these columns.
    scaling : list[str]
        A list of identifiers used to track what scaling was applied to the data, e.g.,
        `"mass"`, `"eicosane_mass"`, `"molecular_weight"`, `"diamagnetic_correction"`.

    Methods
    -------
    scale_moment(mass=0, eicosane_mass=0, molecular_weight=0, diamagnetic_correction=0)
        Scale the moment by the given values. Scaled values are added to the `data`
        attribute in the following columns: `"chi"`, `"chi_err"`, `"chi_t"`,
        `"chi_t_err"`, `"moment"`, and `"moment_err"`. The units of these columns will
        be determined by what scaling was applied.
    simplified_data(*args, **kwargs)
        Returns a simplified version of the data, with only the columns needed for
        most analyses and plotting. These columns are: `"time"` (in seconds),
        `"temperature"` (in Kelvin), `"field"` (in Oe), `"moment"`, `"moment_err"`,
        `"chi"`, `"chi_err"`, `"chi_t"`, and `"chi_t_err"`. Where units are not
        specified, they are determined by what scaling was applied.
    """

    data: pd.DataFrame
    scaling: list[str]

    def scale_moment(
        self,
        mass: float = 0,
        eicosane_mass: float = 0,
        molecular_weight: float = 0,
        diamagnetic_correction: float = 0,
    ) -> None:
        ...

    def simplified_data(self, *args, **kwargs) -> pd.DataFrame:
        ...
