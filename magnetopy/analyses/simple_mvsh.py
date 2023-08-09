from dataclasses import dataclass, asdict
from typing import Any, Literal

import pandas as pd

from magnetopy.magnetometry import Magnetometry
from magnetopy.experiments.mvsh import MvsH


@dataclass
class SimpleMvsHAnalysisParsingArgs:
    """Arguments needed to parse a `Magnetometry` object during the course of an
    analysis performed by `SimpleMvsHAnalysis`.

    Attributes
    ----------
    temperature : float
        The temperature in Kelvin of the measurement to be analyzed.
    segments : Literal["auto", "loop", "forward", "reverse"], optional
        The segments of the measurement to be analyzed. If `"auto"`, the forward and
        reverse segments will be analyzed if they exist and will be ignored if they
        don't. If `"loop"`, the forward and reverse segments will be analyzed if they
        exist and an error will be raised if they don't. If `"forward"` or `"reverse"`,
        only the forward or reverse segment will be analyzed, respectively.
    """

    temperature: float
    segments: Literal["auto", "loop", "forward", "reverse"] = "auto"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SimpleMvsHAnalysisResults:
    """The results of an analysis performed by `SimpleMvsHAnalysis`.

    Attributes
    ----------
    m_s : float
        The saturation magnetization of the sample in units of `moment_units`.
    h_c : float
        The coercive field of the sample in units of `field_units`.
    m_r : float
        The remanent magnetization of the sample in units of `moment_units`.
    moment_units : str
        The units of the saturation magnetization and remanent magnetization.
    field_units : str
        The units of the coercive field.
    segments : list[{"forward", "reverse"}]
        The segments of the measurement that were analyzed.
    """

    m_s: float
    h_c: float
    m_r: float
    moment_units: str
    field_units: str
    segments: Literal["forward", "reverse"]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class SimpleMvsHAnalysis:
    """An analysis of an M vs. H experiment that determines basic information about the
    hysteresis loop.

    Parameters
    ----------
    dataset : Magnetometry
        The `Magnetometry` object which contains the `MvsH` object to be analyzed.
    parsing_args : SimpleMvsHAnalysisParsingArgs
        Arguments needed to parse the `Magnetometry` object to obtain the `MvsH` object
        to be analyzed.

    Attributes
    ----------
    parsing_args : SimpleMvsHAnalysisParsingArgs
        Arguments needed to parse the `Magnetometry` object to obtain the `MvsH` object
        to be analyzed.
    mvsh : MvsH
        The analyzed `MvsH` object.
    results : SimpleMvsHAnalysisResults
        The results of the analysis.
    """

    def __init__(
        self,
        dataset: Magnetometry,
        parsing_args: SimpleMvsHAnalysisParsingArgs,
    ) -> None:
        self.parsing_args = parsing_args
        self.mvsh = dataset.get_mvsh(self.parsing_args.temperature)
        segments = self._get_segments()
        m_s = self._determine_m_s(segments)
        h_c = self._determine_h_c(segments)
        m_r = self._determine_m_r(segments)
        moment_units = self._determine_moment_units()
        field_units = "Oe"
        self.results = SimpleMvsHAnalysisResults(
            m_s, h_c, m_r, moment_units, field_units, list(segments.keys())
        )

    def _get_segments(self) -> dict[str, pd.DataFrame]:
        segments: dict[str : pd.DataFrame] = {}
        if self.parsing_args.segments == "auto":
            try:
                segments["forward"] = self.mvsh.simplified_data("forward")
            except MvsH.SegmentError:
                pass
            try:
                segments["reverse"] = self.mvsh.simplified_data("reverse")
            except MvsH.SegmentError:
                pass
        else:
            if self.parsing_args.segments in ["loop", "forward"]:
                segments["forward"] = self.mvsh.simplified_data("forward")
            if self.parsing_args.segments in ["loop", "reverse"]:
                segments["reverse"] = self.mvsh.simplified_data("reverse")
        return segments

    def _determine_m_s(self, segments: dict[str, pd.DataFrame]) -> float:
        m_s = 0
        for segment in segments.values():
            m_s += (segment["moment"].max() + abs(segment["moment"].min())) / 2
        return m_s / len(segments)

    def _determine_h_c(self, segments: dict[str, pd.DataFrame]) -> float:
        h_c = 0
        for segment in segments.values():
            h_c += abs(segment["field"].iloc[segment["moment"].abs().idxmin()])
        return h_c / len(segments)

    def _determine_m_r(self, segments: dict[str, pd.DataFrame]) -> float:
        m_r = 0
        for segment in segments.values():
            m_r += abs(segment["moment"].iloc[segment["field"].abs().idxmin()])
        return m_r / len(segments)

    def _determine_moment_units(self) -> str:
        scaling = self.mvsh.scaling
        if not scaling:
            return "emu"
        elif "mass" in scaling:
            return "emu/g"
        elif "molar" in scaling:
            return "bohr magnetons/mol"

    def as_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the analysis.

        Returns
        -------
        dict[str, Any]
            Keys are `"mvsh"`, `"parsing_args"`, and `"results"`.
        """
        return {
            "mvsh": self.mvsh,
            "parsing_args": self.parsing_args,
            "results": self.results,
        }
