from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Literal, TypeVar

import pandas as pd
import numpy as np
import numpy.typing as npt
from lmfit import Parameters, minimize, minimizer

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
    """Parameters for a single Cauchy term. All three terms can be floats, two-tuples,
    or three-tuples. If they are floats, they are used as the initial value for the
    fit. If they are two-tuples, the values represent the lower and upper bounds for
    the fit. If they are three-tuples, the values represent the initial value, lower
    bound, and upper bound for the fit.

    All terms are optional. If a term is not provided, it is assumed to be zero.

    Any `float` values passed are cast to tuples.

    Attributes
    ----------
    m_s : float | tuple[float, float] | tuple[float, float, float]
        The saturation magnetization of the term. If a float, the value is used as the
        initial value for the fit. If a two-tuple, the values represent the lower and
        upper bounds for the fit. If a three-tuple, the values represent the initial
        value, lower bound, and upper bound for the fit.
    h_c : float | tuple[float, float] | tuple[float, float, float]
        The coercive field of the term. If a float, the value is used as the initial
        value for the fit. If a two-tuple, the values represent the lower and upper
        bounds for the fit. If a three-tuple, the values represent the initial value,
        lower bound, and upper bound for the fit.
    gamma: float | tuple[float, float] | tuple[float, float, float]
        The term describing the broadness of the term. If a float, the value is used as
        the initial value for the fit. If a two-tuple, the values represent the lower
        and upper bounds for the fit. If a three-tuple, the values represent the initial
        value, lower bound, and upper bound for the fit.
    """

    m_s: float | tuple[float, float] | tuple[
        float, float, float
    ] = 0.0  # either an inital value or (initial_value, min, max)
    h_c: float | tuple[float, float] | tuple[
        float, float, float
    ] = 0.0  # either an inital value or (initial_value, min, max)
    gamma: float | tuple[float, float] | tuple[
        float, float, float
    ] = 0.0  # either an inital value or (initial_value, min, max)

    def __post_init__(self):
        if isinstance(self.m_s, float):
            self.m_s = tuple(self.m_s)
        if isinstance(self.h_c, float):
            self.h_c = tuple(self.h_c)
        if isinstance(self.gamma, float):
            self.gamma = tuple(self.gamma)


@dataclass
class CauchyFittingArgs:
    terms: list[CauchyParams]

    def __post_init__(self):
        self.terms = (
            [self.terms] if isinstance(self.terms, CauchyParams) else self.terms
        )

    def build_params(self, x_range: tuple[float, float], y_sat: float) -> Parameters:
        params = Parameters()
        for i, term in enumerate(self.terms):
            if len(term.m_s) == 1:
                term.m_s = (term.m_s[0], 0, y_sat / len(self.terms))
            elif len(term.m_s) == 2:
                term.m_s = (
                    (term.m_s[0] + term.m_s[1]) / 2,
                    term.m_s[0],
                    term.m_s[1],
                )
            params.add(
                f"m_s_{i}",
                value=term.m_s[0],
                min=term.m_s[1],
                max=term.m_s[2],
            )

            if len(term.h_c) == 1:
                term.h_c = (term.h_c[0], x_range[0], x_range[1])
            elif len(term.h_c) == 2:
                term.h_c = (
                    (term.h_c[0] + term.h_c[1]) / 2,
                    term.h_c[0],
                    term.h_c[1],
                )
            params.add(
                f"h_c_{i}",
                value=term.h_c[0],
                min=term.h_c[1],
                max=term.h_c[2],
            )

            if len(term.gamma) == 1:
                term.gamma = (term.gamma[0], 0, x_range[1] - x_range[0])
            elif len(term.gamma) == 2:
                term.gamma = (
                    (term.gamma[0] + term.gamma[1]) / 2,
                    term.gamma[0],
                    term.gamma[1],
                )
            params.add(
                f"gamma_{i}",
                value=term.gamma[0],
                min=term.gamma[1],
                max=term.gamma[2],
            )
        params.add("chi_pd", value=0)
        return params

    def generate_data(self, x: npt.NDArray) -> npt.NDArray:
        params = Parameters()
        for i, term in enumerate(self.terms):
            params.add(name=f"m_s_{i}", value=term.m_s[0])
            params.add(name=f"h_c_{i}", value=term.h_c[0])
            params.add(name=f"gamma_{i}", value=term.gamma[0])
        params.add(name="chi_pd", value=0)
        return cauchy_pdf(x, params)

    @classmethod
    def build_from_num_terms(
        cls, num_terms: int, x_range: tuple[float, float], y_max: float
    ) -> CauchyFittingArgs:
        terms = []
        for _ in range(num_terms):
            terms.append(
                CauchyParams(
                    m_s=(y_max / num_terms, 0, y_max),
                    h_c=((x_range[0] + x_range[1]) / 2, x_range[0], x_range[1]),
                    gamma=((x_range[1] - x_range[0]) / 10, 0, x_range[1] - x_range[0]),
                )
            )
        return cls(terms)


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
    chi_squared: float
    reduced_chi_squared: float
    m_s_unit: str = "unknown"
    h_c_unit: str = "unknown"
    gamma_unit: str = "unknown"
    chi_pd_unit: str = "unknown"

    def set_units(self, m_s: str, h_c: str, gamma: str, chi_pd: str) -> None:
        self.m_s_unit = m_s
        self.h_c_unit = h_c
        self.gamma_unit = gamma
        self.chi_pd_unit = chi_pd

    def as_dict(self) -> dict[str, Any]:
        output = asdict(self)
        output["_class_"] = self.__class__.__name__
        return output

    def simulate_data(self, x: npt.NDArray) -> npt.NDArray:
        fit_params = Parameters()
        for i, term in enumerate(self.terms):
            fit_params.add(name=f"m_s_{i}", value=term.m_s)
            fit_params.add(name=f"h_c_{i}", value=term.h_c)
            fit_params.add(name=f"gamma_{i}", value=term.gamma)
        fit_params.add(name="chi_pd", value=self.chi_pd)
        return cauchy_pdf(x, fit_params)


class CauchyPDFAnalysis:
    def __init__(
        self,
        dataset: Magnetometry,
        parsing_args: CauchyParsingArgs,
        fitting_args: CauchyFittingArgs | int,
    ) -> None:
        self.parsing_args = parsing_args
        self.fitting_args = fitting_args

        data = self.parsing_args.prepare_data(dataset, "pdf")

        x_range = (np.min(data["h"]), np.max(data["h"]))
        y_max = np.max(abs(data["target"]))
        if isinstance(fitting_args, int):
            fitting_args = CauchyFittingArgs.build_from_num_terms(
                fitting_args, x_range, y_max
            )
        params = fitting_args.build_params(x_range, y_max)

        self.results = fit_cauchy_pdf(data["h"], data["target"], params)


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


def fit_cauchy_pdf(
    x: npt.NDArray, y: npt.NDArray, fitting_args: CauchyFittingArgs | int
) -> CauchyAnalysisResults:
    x_range = (np.min(x), np.max(x))
    integral_y = np.cumsum(y * np.gradient(x))
    y_sat = (integral_y.max() - integral_y.min()) / 2
    if isinstance(fitting_args, int):
        fitting_args = CauchyFittingArgs.build_from_num_terms(
            fitting_args, x_range, y_sat
        )
    params = fitting_args.build_params(x_range, y_sat)

    def residual(params: Parameters, h: float, dmdh_data: float) -> float:
        return cauchy_pdf(h, params) - dmdh_data

    lmfit_results: minimizer.MinimizerResult = minimize(
        residual,
        params,
        args=(x, y),
    )

    results = _build_results(lmfit_results, len(params) // 3)

    return results


def _build_results(
    results: minimizer.MinimizerResult,
    num_terms: int,
) -> CauchyAnalysisResults:
    terms: list[CauchyTermResults] = []
    for i in range(num_terms):
        terms.append(
            CauchyTermResults(
                m_s=results.params[f"m_s_{i}"].value,
                m_s_err=results.params[f"m_s_{i}"].stderr,
                h_c=results.params[f"h_c_{i}"].value,
                h_c_err=results.params[f"h_c_{i}"].stderr,
                gamma=results.params[f"gamma_{i}"].value,
                gamma_err=results.params[f"gamma_{i}"].stderr,
            )
        )
    return CauchyAnalysisResults(
        terms=terms,
        chi_pd=results.params["chi_pd"].value,
        chi_pd_err=results.params["chi_pd"].stderr,
        chi_squared=results.chisqr,
        reduced_chi_squared=results.redchi,
    )
