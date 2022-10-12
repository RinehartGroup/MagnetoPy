from enum import Enum, auto
from typing import Tuple

import pandas as pd

from magnetopy.parse_qd import *


class Measurement(Enum):
    FC = auto()
    ZFC = auto()
    MVSH = auto()


class Sample:
    def __init__(self, files: list[Tuple[QDFile, Measurement]], **kwargs):

        self.dataset = pd.DataFrame(
            {
                "Measurement Type": [file[1] for file in files],
                "File": [file[0] for file in files],
            }
        )
        options = {
            "samplt_wt": None,
            "sample_vol": None,
            "matrix": None,
            "matrix_wt": None,
            "matrix_vol": None,
        }

        # something for diamagnetic correction
        #


class DiamagneticCorrection:
    def __init__(self, material: str, amount: float):
        pass
