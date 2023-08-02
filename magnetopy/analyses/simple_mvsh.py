from dataclasses import dataclass

import pandas as pd

from magnetopy.dataset import Dataset
from magnetopy.experiments import MvsH


@dataclass
class SimpleMvsHAnalysisParsingArgs:
    temperature: float
    segments: str = "auto"  # auto, loop, forward, reverse


@dataclass
class SimpleMvsHAnalysisResults:
    m_s: float
    h_c: float
    m_r: float
    segments: list[str]


class SimpleMvsHAnalysis:
    def __init__(
        self,
        dataset: Dataset,
        parsing_args: SimpleMvsHAnalysisParsingArgs,
    ) -> None:
        self.parsing_args = parsing_args
        self.mvsh = dataset.get_mvsh(self.parsing_args.temperature)
        self.segments = self._get_segments()
        m_s = self._determine_m_s()
        h_c = self._determine_h_c()
        m_r = self._determine_m_r()
        self.results = SimpleMvsHAnalysisResults(
            m_s, h_c, m_r, list(self.segments.keys())
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

    def _determine_m_s(self) -> float:
        m_s = 0
        for segment in self.segments.values():
            m_s += (segment["moment"].max() + abs(segment["moment"].min())) / 2
        return m_s / len(self.segments)

    def _determine_h_c(self) -> float:
        h_c = 0
        for segment in self.segments.values():
            h_c += abs(segment["field"].iloc[segment["moment"].abs().idxmin()])
        return h_c / len(self.segments)

    def _determine_m_r(self) -> float:
        m_r = 0
        for segment in self.segments.values():
            m_r += abs(segment["moment"].iloc[segment["field"].abs().idxmin()])
        return m_r / len(self.segments)
