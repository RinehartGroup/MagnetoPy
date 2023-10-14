from dataclasses import dataclass, asdict
from typing import Any, Literal, TypeVar

import pandas as pd
import numpy as np
import numpy.typing as npt
from lmfit import Parameters, minimize

from magnetopy.magnetometry import Magnetometry
from magnetopy.experiments.mvsh import MvsH

FitVal = TypeVar("FitVal", float, npt.NDArray)


@dataclass
class CauchyParsingArgs:
    """Arguments needed to parse a `Magnetometry` object during the course of an
    analysis performed by `CauchyPDFAnalysis` or `CauchyCDFAnalysis`.

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
    experiment : Literal["MvsH"], optional
        The type of measurement to be analyzed. Reserved for future use (e.g., `"ACvsH"`
        experiments). Defaults to `"MvsH"`.
    """

    temperature: float
    segments: Literal["auto", "loop", "forward", "reverse"] = "auto"
    experiment: Literal["MvsH"] = "MvsH"

    class InvalidExperimentError(Exception):
        """Raised when the experiment is not supported by the analysis."""

    def as_dict(self) -> dict[str, Any]:
        output = asdict(self)
        output["_class_"] = self.__class__.__name__
        return output

    def prepare_data(
        self, dataset: Magnetometry, analysis: Literal["cdf", "pdf"]
    ) -> pd.DataFrame:
        """Parses the given dataset and returns a DataFrame containing the data to be
        analyzed.

        Parameters
        ----------
        dataset : Magnetometry
            The Magnetometry object containing the data to be analyzed.
        analysis : Literal["cdf", "pdf"]
            The type of analysis to be performed.

        Returns
        -------
        pd.DataFrame
            DataFrame has two columns: "h" and "target", where "h" is the applied field
            and "target" is the value under investigation, e.g., the magnetization (M),
            the derivative of magnetization with respect to field (dM/dH),
            or a form of susceptibility (e.g., chi, chi', chi").

        Raises
        ------
        InvalidExperimentError
            If the experiment is not supported by the analysis.
        """
        if self.experiment == "MvsH":
            return self._prepare_mvsh_data(dataset, analysis)
        raise self.InvalidExperimentError(
            f"Experiment {self.experiment} is not supported by the analysis."
        )

    def _prepare_mvsh_data(
        self, dataset: Magnetometry, analysis: Literal["cdf", "pdf"]
    ) -> pd.DataFrame:
        mvsh = dataset.get_mvsh(self.temperature)
        data = pd.DataFrame()

        def _get_segment_data(segment: str):
            # uses `analysis` and `mvsh` from the outer scope
            df = pd.DataFrame()
            df["h"] = mvsh.simplified_data(segment)["field"]
            if analysis == "cdf":
                df["target"] = mvsh.simplified_data(segment)["moment"]
            else:
                df["target"] = np.gradient(
                    mvsh.simplified_data(segment)["moment"],
                    mvsh.simplified_data(segment)["field"],
                )
            return df

        if self.segments == "auto":
            try:
                temp = _get_segment_data("forward")
                data = pd.concat([data, temp])
            except MvsH.SegmentError:
                pass
            try:
                temp = _get_segment_data("reverse")
                data = pd.concat([data, temp])
            except MvsH.SegmentError:
                pass
        else:
            if self.segments in ["loop", "forward"]:
                temp = _get_segment_data("forward")
                data = pd.concat([data, temp])
            if self.segments in ["loop", "reverse"]:
                temp = _get_segment_data("reverse")
                data = pd.concat([data, temp])

        return data

    def get_units(self):
        pass


@dataclass
class CauchyParams:
    m_s: float | tuple[
        float, float, float
    ] = 0.0  # either an inital value or (initial_value, min, max)
    h_c: float | tuple[float, float, float] = 0.0
    gamma: float | tuple[float, float, float] = 0.0

    def rewrite_as_tuples(self):
        if isinstance(self.m_s, float | int):
            self.m_s = (self.m_s, 0, 1)
        elif isinstance(self.m_s, tuple):
            if len(self.m_s) != 3:
                raise ValueError("m_s must be a tuple of length 3")
        if isinstance(self.h_c, float | int):
            self.h_c = (self.h_c, -700000, 700000)
        elif isinstance(self.h_c, tuple):
            if len(self.h_c) != 3:
                raise ValueError("h_c must be a tuple of length 3")
        if isinstance(self.gamma, float | int):
            self.gamma = (self.gamma, 0, 100000)


@dataclass
class CauchyFittingArgs:
    cauchy_params: list[CauchyParams]

    def __post_init__(self):
        self.cauchy_params = (
            [self.cauchy_params]
            if isinstance(self.cauchy_params, CauchyParams)
            else self.cauchy_params
        )

    def build_params(self) -> Parameters:
        params = Parameters()
        for i, cauchy_param in enumerate(self.cauchy_params):
            cauchy_param.rewrite_as_tuples()
            params.add(
                f"m_s_{i}",
                value=cauchy_param.m_s[0],
                min=cauchy_param.m_s[1],
                max=cauchy_param.m_s[2],
            )
            params.add(
                f"h_c_{i}",
                value=cauchy_param.h_c[0],
                min=cauchy_param.h_c[1],
                max=cauchy_param.h_c[2],
            )
            params.add(
                f"gamma_{i}",
                value=cauchy_param.gamma[0],
                min=cauchy_param.gamma[1],
                max=cauchy_param.gamma[2],
            )
        params.add("chi_pd", value=0, min=-1, max=1)
        return params


@dataclass
class CauchyTermResults:
    m_s: float
    m_s_err: float
    h_c: float
    h_c_err: float
    gamma: float
    gamma_err: float


@dataclass
class CauchyAnalysisResults:
    terms: list[CauchyTermResults]
    chi_pd: float
    chi_pd_err: float
    error: float
    m_s_unit: str
    h_c_unit: str
    gamma_unit: str
    chi_pd_unit: str


class CauchyPDFAnalysis:
    def __init__(
        self,
        dataset: Magnetometry,
        parsing_args: CauchyParsingArgs,
        fitting_args: CauchyFittingArgs,
    ) -> None:
        self.parsing_args = parsing_args
        self.fitting_args = fitting_args

        params = self.fitting_args.build_params()
        data = self.parsing_args.prepare_data(dataset, "pdf")

        def residual(params: Parameters, h: float, dmdh_data: float) -> float:
            return self.cauchy_pdf(h, params) - dmdh_data

        results = minimize(
            residual,
            params,
            args=(data["h"], data["target"]),
        )

    @staticmethod
    def cauchy_pdf(h: FitVal, params: Parameters) -> FitVal:
        # `params` contains
        #   [
        #       m_s_0, h_c_0, gamma_0,
        #       m_s_1, h_c_1, gamma_1,
        #       ...
        #       chi_pd
        #   ]
        try:
            m = params["chi_pd"]
        except KeyError:
            m = 0
        for i in range(0, len(params) - 1, 3):
            j = i // 3
            m += (8 * params[f"m_s_{j}"] * params[f"gamma_{j}"]) / (
                np.pi * (16 * (h - params[f"h_c_{j}"]) ** 2 + params[f"gamma_{j}"] ** 2)
            )
        return m
