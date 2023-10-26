from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Literal, TypeVar

import numpy as np
import numpy.typing as npt
import pandas as pd
from lmfit import Parameters, minimize, minimizer

FitVal = TypeVar("FitVal", float, npt.ArrayLike)


def cauchy_pdf(h: FitVal, params: Parameters) -> FitVal:
    """Generates data representing a sum of Cauchy probability density functions
    (PDFs).

    Equation:

    ```math
    \\frac{dM}{dH}(H, M_s, H_c, \\gamma) = \\chi_{pd} + \\
    \\sum_1^n \\frac{8M_{s,n}}{\\pi} \\frac{\\gamma_n}{16 (H-H_{c,n})^2 + \\
    \\gamma_n^2 }
    ```

    Parameters
    ----------
    h : FitVal
        A FitVal (either a float or a numpy array) representing the x data (e.g.,
        magnetic field).
    params : Parameters
        Params must contain the following parameters as a valid `lmfit.Parameters`
        object:
            - `m_s_0`, `h_c_0`, `gamma_0`
            - `m_s_1`, `h_c_1`, `gamma_1`
            - ...
            - `chi_pd`

    Returns
    -------
    FitVal
        The float or numpy array representing the y data (e.g., the derivative of
        magnetization with respect to field).
    """
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
    x: npt.ArrayLike, y: npt.ArrayLike, fitting_args: CauchyFittingArgs | int
) -> CauchyAnalysisResults:
    """Fits given x and y data (e.g., magnetic field and the derivative of
    magnetization with respect to field) to a sum of Cauchy probability density
    functions (PDFs).

    Parameters
    ----------
    x : npt.ArrayLike
        The x data, e.g., magnetic field.
    y : npt.ArrayLike
        The y data, e.g., the derivative of magnetization with respect to field.
    fitting_args : CauchyFittingArgs | int
        The arguments needed to perform the fit. If an `int` is given, the number of
        terms to be used in the fit. If a `CauchyFittingArgs` object is given, the
        parameters to be used in the fit.

    Returns
    -------
    CauchyAnalysisResults
        The results of the fit.
    """
    x_range = (np.min(x), np.max(x))
    integral_y: npt.NDArray = np.cumsum(y * np.gradient(x))
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


def cauchy_cdf(h: FitVal, params: Parameters) -> FitVal:
    """Generates data representing a sum of Cauchy cumulative distribution functions
    (CDFs).

    Equation:

    ```math
    M(H, M_s, H_c, \\gamma) = \\chi_{pd}H + \\sum_1^n \\frac{2M_{s,n}}{\\pi}
    \\arctan\\left(\\frac{H-H_{c,n}}{\\gamma_n}\\right)
    ```

    Parameters
    ----------
    h : FitVal
        A FitVal (either a float or a numpy array) representing the x data (e.g.,
        magnetic field).
    params : Parameters
        Params must contain the following parameters as a valid `lmfit.Parameters`
        object:
            - `m_s_0`, `h_c_0`, `gamma_0`
            - `m_s_1`, `h_c_1`, `gamma_1`
            - ...
            - `chi_pd`

    Returns
    -------
    FitVal
        The float or numpy array representing the y data (e.g., the magnetization with
        respect to field).
    """
    try:
        m = h * params["chi_pd"]
    except KeyError:
        m = 0
    for i in range(0, len(params) - 1, 3):
        j = i // 3
        m += (2 * params[f"m_s_{j}"] / np.pi) * np.arctan(
            (h - params[f"h_c_{j}"]) / params[f"gamma_{j}"]
        )
    return m


def fit_cauchy_cdf(
    x: npt.ArrayLike, y: npt.ArrayLike, fitting_args: CauchyFittingArgs | int
) -> CauchyAnalysisResults:
    """Fits given x and y data (e.g., magnetic field and magnetization) to a sum of
    Cauchy cumulative distribution functions (CDFs).

    Parameters
    ----------
    x : npt.ArrayLike
        The x data, e.g., magnetic field.
    y : npt.ArrayLike
        The y data, e.g., the magnetization.
    fitting_args : CauchyFittingArgs | int
        The arguments needed to perform the fit. If an `int` is given, the number of
        terms to be used in the fit. If a `CauchyFittingArgs` object is given, the
        parameters to be used in the fit.

    Returns
    -------
    CauchyAnalysisResults
        The results of the fit.
    """
    x_range = (np.min(x), np.max(x))
    y_sat = np.max(y)
    if isinstance(fitting_args, int):
        fitting_args = CauchyFittingArgs.build_from_num_terms(
            fitting_args, x_range, y_sat
        )
    params = fitting_args.build_params(x_range, y_sat)

    def residual(params: Parameters, h: float, data: float) -> float:
        return cauchy_cdf(h, params) - data

    lmfit_results: minimizer.MinimizerResult = minimize(
        residual,
        params,
        args=(x, y),
    )

    results = _build_results(lmfit_results, len(params) // 3)

    return results


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
        if isinstance(self.m_s, float | int):
            self.m_s = tuple([self.m_s])
        if isinstance(self.h_c, float | int):
            self.h_c = tuple([self.h_c])
        if isinstance(self.gamma, float | int):
            self.gamma = tuple([self.gamma])

    def as_dict(self) -> dict[str, Any]:
        """Returns a dictionary representation of the object.

        Returns
        -------
        dict[str, Any]
            Keys are: "_class_", "m_s", "h_c", "gamma".
        """
        output = {"_class_": self.__class__.__name__}
        output.update(asdict(self))
        return output


@dataclass
class CauchyFittingArgs:
    """_summary_

    Attributes
    ----------
    terms : list[CauchyParams] | CauchyParams
        The parameters for each individual Cauchy term to be used in the fit. A single
        term will be converted to a list of length one.
    """

    terms: list[CauchyParams]

    def __post_init__(self):
        self.terms = (
            [self.terms] if isinstance(self.terms, CauchyParams) else self.terms
        )

    def build_params(self, x_range: tuple[float, float], y_sat: float) -> Parameters:
        """Uses the `CauchyParams` given during object creation to build a `Parameters`
        object compatible with `lmfit`.

        Parameters
        ----------
        x_range : tuple[float, float]
            The minimum and maximum values of the x-axis.
        y_sat : float
            In terms of magnetization, this is the saturation magnetization of the
            original M vs. H data. If the data under consideration is the derivative of
            magnetization with respect to field or is otherwise data in a form similar
            to that of the Cauchy PDF, y_sat will be half of the integral of the data
            over the x_range.

        Returns
        -------
        Parameters
            A `Parameters` object compatible with `lmfit`. There will be three parameters
            (`m_s`, `h_c`, and `gamma`) for each term in `self.terms` and one parameter
            (`chi_pd`) to account for a phenomena whose contribution to the magnetization
            is linear with respect to field (i.e., paramagnetism and diamagnetism).
        """
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

    def generate_data(
        self, x: npt.ArrayLike, form: Literal["pdf", "cdf"]
    ) -> npt.ArrayLike:
        """Generates simulated data representing a sum of Cauchy PDFs using the
        individual terms given during object creation.

        Parameters
        ----------
        x : npt.ArrayLike
            The x data, e.g., magnetic field.
        form : {"pdf", "cdf"}
            The form of the Cauchy distribution to generate.

        Returns
        -------
        npt.ArrayLike
            The y data, e.g., the derivative of magnetization with respect to field.
        """
        params = Parameters()
        for i, term in enumerate(self.terms):
            params.add(name=f"m_s_{i}", value=term.m_s[0])
            params.add(name=f"h_c_{i}", value=term.h_c[0])
            params.add(name=f"gamma_{i}", value=term.gamma[0])
        params.add(name="chi_pd", value=0)
        if form.lower() == "cdf":
            simulated_data = cauchy_cdf(x, params)
        elif form.lower() == "pdf":
            simulated_data = cauchy_pdf(x, params)
        else:
            raise ValueError(f"`form` argument must be 'pdf' or 'cdf', not {form}")
        return simulated_data

    def as_dict(self) -> dict[str, Any]:
        """Returns a dictionary representation of the object.

        Returns
        -------
        dict[str, Any]
            Keys are: "_class_", "terms".
        """
        output = {"_class_": self.__class__.__name__}
        output["terms"] = [term.as_dict() for term in self.terms]
        return output

    @classmethod
    def build_from_num_terms(
        cls, num_terms: int, x_range: tuple[float, float], y_sat: float
    ) -> CauchyFittingArgs:
        """Generates input parameters for a Cauchy fit based on the number of terms
        desired.

        Parameters
        ----------
        num_terms : int
            The number of Cauchy terms to be used in the fit.
        x_range : tuple[float, float]
            The minimum and maximum values of the x-axis.
        y_sat : float
            In terms of magnetization, this is the saturation magnetization of the
            original M vs. H data. If the data under consideration is the derivative of
            magnetization with respect to field or is otherwise data in a form similar
            to that of the Cauchy PDF, y_sat will be half of the integral of the data
            over the x_range.

        Returns
        -------
        CauchyFittingArgs
            The input parameters for a Cauchy fit.
        """
        terms = []
        y_sat_padded = 1.5 * y_sat
        for _ in range(num_terms):
            terms.append(
                CauchyParams(
                    m_s=(y_sat / num_terms, 0, y_sat_padded),
                    h_c=((x_range[0] + x_range[1]) / 2, x_range[0], x_range[1]),
                    gamma=((x_range[1] - x_range[0]) / 10, 0, x_range[1] - x_range[0]),
                )
            )
        return cls(terms)


@dataclass
class CauchyTermResults:
    """The fit results for a single Cauchy term. Note that these terms assume that the
    data was either M vs. H or dM/dH vs. H.

    Attributes
    ----------
    m_s : float
        The saturation magnetization of the term.
    m_s_err : float
        The error in the saturation magnetization of the term.
    h_c : float
        The coercive field of the term.
    h_c_err : float
        The error in the coercive field of the term.
    gamma : float
        The parameter describing the broadness of the term.
    gamma_err : float
        The error in the parameter describing the broadness of the term.
    """

    m_s: float
    m_s_err: float
    h_c: float
    h_c_err: float
    gamma: float
    gamma_err: float


@dataclass
class CauchyAnalysisResults:
    """The results of a fit to a Cauchy analysis (either CDF or PDF).

    Attributes
    ----------
    terms : list[CauchyTermResults]
        The fit results for each individual Cauchy term; each term includes values for
        `m_s`, `h_c`, `gamma`, and their errors.
    chi_pd : float
        The fit value for the term describing a phenomena whose contribution to the
        magnetization is linear with respect to field (i.e., paramagnetism and
        diamagnetism).
    chi_pd_err : float
        The error of `chi_pd`.
    chi_squared : float
        The chi-squared value of the fit.
    reduced_chi_squared : float
        The reduced chi-squared value of the fit.
    m_s_unit : str
        The units of the `m_s`.
    h_c_unit : str
        The units of the `h_c`.
    gamma_unit : str
        The units of `gamma`.
    chi_pd_unit : str
        The units of `chi_pd`.
    """

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
        """Sets the units of the fit results.

        Parameters
        ----------
        m_s : str
            The units of the `m_s`.
        h_c : str
            The units of the `h_c`.
        gamma : str
            The units of `gamma`.
        chi_pd : str
            The units of `chi_pd`.
        """
        self.m_s_unit = m_s
        self.h_c_unit = h_c
        self.gamma_unit = gamma
        self.chi_pd_unit = chi_pd

    def as_dict(self) -> dict[str, Any]:
        """Returns a dictionary representation of the object.

        Returns
        -------
        dict[str, Any]
            Keys are: "_class_", "terms", "chi_pd", "chi_pd_err", "chi_squared",
            "reduced_chi_squared", "m_s_unit", "h_c_unit", "gamma_unit", "chi_pd_unit".
        """
        output = {"_class_": self.__class__.__name__}
        output.update(asdict(self))
        return output

    def generate_data(
        self, x: npt.ArrayLike, form: Literal["pdf", "cdf"]
    ) -> npt.ArrayLike:
        """Generates simulated data representing a sum of Cauchy PDFs or CDFs using the
        individual terms resulting from the fit.

        Parameters
        ----------
        x : npt.ArrayLike
            The x data, e.g., magnetic field.
        form : {"pdf", "cdf"}
            The form of the Cauchy distribution to generate.

        Returns
        -------
        npt.ArrayLike
            The y data, e.g., the derivative of magnetization with respect to field.
        """
        params = Parameters()
        for i, term in enumerate(self.terms):
            params.add(name=f"m_s_{i}", value=term.m_s)
            params.add(name=f"h_c_{i}", value=term.h_c)
            params.add(name=f"gamma_{i}", value=term.gamma)
        params.add(name="chi_pd", value=self.chi_pd)
        if form.lower() == "cdf":
            simulated_data = cauchy_cdf(x, params)
        elif form.lower() == "pdf":
            simulated_data = cauchy_pdf(x, params)
        else:
            raise ValueError(f"`form` argument must be 'pdf' or 'cdf', not {form}")
        return simulated_data

    def generate_data_by_term(
        self, x: npt.ArrayLike, form: Literal["pdf", "cdf"]
    ) -> list[npt.ArrayLike]:
        """Generates simulated data representing each individual term resulting from the
        fit.

        Parameters
        ----------
        x : npt.ArrayLike
            The x data, e.g., magnetic field.
        form : {"pdf", "cdf"}
            The form of the Cauchy distribution to generate.

        Returns
        -------
        list[npt.ArrayLike]
            The y data, e.g., the derivative of magnetization with respect to field.
        """
        simulated_data = []
        for term in self.terms:
            params = Parameters()
            params.add(name="m_s_0", value=term.m_s)
            params.add(name="h_c_0", value=term.h_c)
            params.add(name="gamma_0", value=term.gamma)
            if form.lower() == "cdf":
                simulated_data.append(cauchy_cdf(x, params))
            elif form.lower() == "pdf":
                simulated_data.append(cauchy_pdf(x, params))
            else:
                raise ValueError(f"`form` argument must be 'pdf' or 'cdf', not {form}")
        return simulated_data


def reverse_sequence(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        try:
            df[column] = df[column] * -1
        except KeyError:
            continue
    return df


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
