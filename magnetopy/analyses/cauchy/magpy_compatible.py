from typing import Any, Literal
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from magnetopy.magnetometry import Magnetometry
from magnetopy.experiments.mvsh import MvsH
from magnetopy.analyses.cauchy.standalone import fit_cauchy_pdf, CauchyFittingArgs
from magnetopy.analyses.cauchy.plots import plot_cauchy_pdf


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
                temp = self.reverse_sequence(temp, ["h"])
                data = pd.concat([data, temp])
            except MvsH.SegmentError:
                pass
        else:
            if self.segments in ["loop", "forward"]:
                temp = _get_segment_data("forward")
                data = pd.concat([data, temp])
            if self.segments in ["loop", "reverse"]:
                temp = _get_segment_data("reverse")
                temp = self.reverse_sequence(temp, ["h"])
                data = pd.concat([data, temp])

        return data

    @staticmethod
    def reverse_sequence(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        df = df.copy()
        for column in columns:
            try:
                df[column] = df[column] * -1
            except KeyError:
                continue
        return df

    def get_units(self):
        pass


class CauchyPDFAnalysis:
    """An analysis of dM/dH vs. H data using a sum of Cauchy probability density
    functions (PDFs).

    Equation:

    $$
    \\frac{dM}{dH}(H, M_s, H_c, \\gamma) = \\chi_{pd} + \\
    \\sum_1^n \\frac{8M_{s,n}}{\\pi} \\frac{\\gamma_n}{16 (H-H_{c,n})^2 + \\
    \\gamma_n^2 }
    $$

    Parameters
    ----------
    dataset : Magnetometry
        The Magnetometry object containing the data to be analyzed.
    parsing_args : CauchyParsingArgs
        The arguments needed to parse the dataset and obtain the data to be analyzed.
    fitting_args : CauchyFittingArgs | int
        The arguments needed to perform the fit. If an `int` is given, the number of
        terms to be used in the fit. If a `CauchyFittingArgs` object is given, the
        parameters to be used in the fit.

    Attributes
    ----------
    parsing_args : CauchyParsingArgs
        The arguments needed to parse the dataset and obtain the data to be analyzed.
    fitting_args : CauchyFittingArgs
        The arguments needed to perform the fit.
    results : CauchyAnalysisResults
        The results of the fit.
    """

    def __init__(
        self,
        dataset: Magnetometry,
        parsing_args: CauchyParsingArgs,
        fitting_args: CauchyFittingArgs | int,
    ) -> None:
        self.parsing_args = parsing_args

        self.data = self.parsing_args.prepare_data(dataset, "pdf")
        self.data.sort_values(by="h", inplace=True)

        self.fitting_args = fitting_args
        x_range = (np.min(self.data["h"]), np.max(self.data["h"]))
        integral_y = np.cumsum(self.data["target"] * np.gradient(self.data["h"]))
        y_sat = (integral_y.max() - integral_y.min()) / 2
        if isinstance(self.fitting_args, int):
            self.fitting_args = CauchyFittingArgs.build_from_num_terms(
                self.fitting_args, x_range, y_sat
            )

        self.results = fit_cauchy_pdf(
            self.data["h"], self.data["target"], self.fitting_args
        )

    def plot(
        self,
        show_full_fit: bool = True,
        show_fit_components: bool = False,
        input_params: bool = False,
        **kwargs,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plots the results of a Cauchy PDF analysis.

        Parameters
        ----------
        x : npt.ArrayLike
            The x data, e.g., magnetic field.
        y : npt.ArrayLike
            They y data, e.g., derivative of magnetization with respect to magnetic field.
        show_full_fit : bool, optional
            Whether to show the full fit, by default True.
        show_fit_components : bool, optional
            Whether to show the individual fit components, by default False.
        input_params : CauchyFittingArgs | None, optional
            If given, plots the terms given by the CauchyFittingArgs object, by default
            None.

        Returns
        -------
        tuple[plt.Figure, plt.Axes]
            The figure and axes objects.
        """
        if input_params:
            input_params = self.fitting_args
        fig, ax = plot_cauchy_pdf(
            self.data["h"],
            self.data["target"],
            results=self.results,
            show_full_fit=show_full_fit,
            show_fit_components=show_fit_components,
            input_params=input_params,
            **kwargs,
        )
        return fig, ax

    def as_dict(self) -> dict[str, Any]:
        """Returns a dictionary representation of the object.

        Returns
        -------
        dict[str, Any]
            Keys are:
        """
        return {
            "_class_": self.__class__.__name__,
            "parsing_args": self.parsing_args,
            "fitting_args": self.fitting_args,
            "results": self.results,
        }
