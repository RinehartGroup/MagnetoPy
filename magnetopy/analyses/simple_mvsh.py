from dataclasses import dataclass, asdict
from typing import Any

import pandas as pd

from magnetopy.dataset import Dataset
from magnetopy.experiments import MvsH


@dataclass
class SimpleMvsHAnalysisParsingArgs:
    temperature: float
    segments: str = "auto"  # auto, loop, forward, reverse

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SimpleMvsHAnalysisResults:
    m_s: float
    h_c: float
    m_r: float
    moment_units: str
    field_units: str
    segments: list[str]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class SimpleMvsHAnalysis:
    def __init__(
        self,
        dataset: Dataset,
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
        return {
            "mvsh": self.mvsh,
            "parsing_args": self.parsing_args,
            "results": self.results,
        }
