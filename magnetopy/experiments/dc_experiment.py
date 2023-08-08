from typing import Protocol

import pandas as pd


class DcExperiment(Protocol):
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
