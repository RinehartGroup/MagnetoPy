from typing import Literal
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from magnetopy.analyses.cauchy.standalone import (
    CauchyFittingArgs,
    CauchyAnalysisResults,
)
from magnetopy.plot_utils import (
    handle_kwargs,
    handle_options,
)
from magnetopy.plot_utils import force_aspect


def plot_cauchy_cdf(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    results: CauchyAnalysisResults | None = None,
    add_reversed_data: bool = False,
    add_reversed_simulated: bool = False,
    show_full_fit: bool = True,
    show_fit_components: bool = False,
    input_params: CauchyFittingArgs | None = None,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """Plots the results of a Cauchy PDF analysis.

    Parameters
    ----------
    x : npt.ArrayLike
        The x data, e.g., magnetic field.
    y : npt.ArrayLike
        They y data, e.g., derivative of magnetization with respect to magnetic field.
    results : CauchyAnalysisResults | None, optional
        The results of the fit, by default None.
    add_reversed_data : bool, optional
        Option to plot the reversed data (e.g., if the data from a forward sweep of an
        M vs H measurement is given, this option will additionally plot the data as if
        it were from a reverse sweep).
    add_reversed_simulated : bool, optional
        Plots the reversed generated data (generated from either/both the input data
        and the simulated data from the fit), by default False.
    show_full_fit : bool, optional
        Assuming results is not None, whether to show the full fit (i.e., the fit
        resulting from the sum of separate Cauchy terms), by default True.
    show_fit_components : bool, optional
        Assuming results is not None, whether to show the individual fit components,
        by default False.
    input_params : CauchyFittingArgs | None, optional
        If given, plots the terms given by the CauchyFittingArgs object, by default
        None.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        The figure and axes objects.
    """
    return plot_cauchy(
        x,
        y,
        "cdf",
        results,
        add_reversed_data,
        add_reversed_simulated,
        show_full_fit,
        show_fit_components,
        input_params,
        **kwargs,
    )


def plot_cauchy_pdf(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    results: CauchyAnalysisResults | None = None,
    show_full_fit: bool = True,
    show_fit_components: bool = False,
    input_params: CauchyFittingArgs | None = None,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """Plots the results of a Cauchy PDF analysis.

    Parameters
    ----------
    x : npt.ArrayLike
        The x data, e.g., magnetic field.
    y : npt.ArrayLike
        They y data, e.g., derivative of magnetization with respect to magnetic field.
    results : CauchyAnalysisResults | None, optional
        The results of the fit, by default None.
    show_full_fit : bool, optional
        Assuming results is not None, whether to show the full fit (i.e., the fit
        resulting from the sum of separate Cauchy terms), by default True.
    show_fit_components : bool, optional
        Assuming results is not None, whether to show the individual fit components,
        by default False.
    input_params : CauchyFittingArgs | None, optional
        If given, plots the terms given by the CauchyFittingArgs object, by default
        None.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        The figure and axes objects.
    """
    return plot_cauchy(
        x,
        y,
        "pdf",
        results,
        show_full_fit=show_full_fit,
        show_fit_components=show_fit_components,
        input_params=input_params,
        **kwargs,
    )


def plot_cauchy(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    form: Literal["pdf", "cdf"],
    results: CauchyAnalysisResults | None = None,
    add_reversed_data: bool = False,
    add_reversed_simulated: bool = False,
    show_full_fit: bool = True,
    show_fit_components: bool = False,
    input_params: CauchyFittingArgs | None = None,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """Plots the results of a Cauchy analysis.

    Parameters
    ----------
    x : npt.ArrayLike
        The x data, e.g., magnetic field.
    y : npt.ArrayLike
        They y data, e.g., derivative of magnetization with respect to magnetic field.
    form : Literal["pdf", "cdf"]
        Whether the data is a probability density function or a cumulative distribution
    results : CauchyAnalysisResults | None, optional
        The results of the fit, by default None.
    add_reversed_data : bool, optional
        For CDF plots, option to plot the reversed data (e.g., if the data from a
        forward sweep of an M vs H measurement is given, this option will additionally
        plot the data as if it were from a reverse sweep).
    add_reversed_simulated : bool, optional
        Plots the reversed generated data (generated from either/both the input data
        and the simulated data from the fit), by default False.
    show_full_fit : bool, optional
        Assuming results is not None, whether to show the full fit (i.e., the fit
        resulting from the sum of separate Cauchy terms), by default True.
    show_fit_components : bool, optional
        Assuming results is not None, whether to show the individual fit components,
        by default False.
    input_params : CauchyFittingArgs | None, optional
        If given, plots the terms given by the CauchyFittingArgs object, by default
        None.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        The figure and axes objects.
    """
    options = handle_kwargs(**kwargs)

    max_h_c = None

    fig, ax = plt.subplots(figsize=options["figsize"])
    x_ = x
    y_ = y
    if add_reversed_data:
        x_ = np.concatenate([x, -1 * x])
        y_ = np.concatenate([y, -1 * y])
    ax.plot(x_, y_, label="Data", color="black")
    if input_params:
        x_input: npt.NDArray = np.linspace(min(x), max(x), 1000)
        y_input: npt.NDArray = input_params.generate_data(x_input, form)
        if add_reversed_simulated:
            x_input = np.concatenate([x_input, -1 * x_input])
            y_input = np.concatenate([y_input, -1 * y_input])
        ax.plot(x_input, y_input, label="Input", color="blue")
        max_h_c = np.max([term.h_c for term in input_params.terms])

    if results:
        x_fit = np.linspace(min(x), max(x), 1000)
        max_h_c = np.max([term.h_c for term in results.terms])
        if show_full_fit:
            x_fit_ = x_fit
            y_fit = results.generate_data(x_fit, form)
            if add_reversed_simulated:
                x_fit_ = np.concatenate([x_fit, -1 * x_fit])
                y_fit = np.concatenate([y_fit, -1 * y_fit])
            ax.plot(x_fit_, y_fit, label="Fit", color="red")
        if show_fit_components:
            x_fit_ = x_fit
            y_fits = results.generate_data_by_term(x_fit, form)
            for i, y_fit in enumerate(y_fits):
                if add_reversed_simulated:
                    x_fit_ = np.concatenate([x_fit, -1 * x_fit])
                    y_fit = np.concatenate([y_fit, -1 * y_fit])
                ax.plot(x_fit_, y_fit, label=f"Fit Term {i + 1}")

    if options["xlabel"]:
        ax.set_xlabel(options["xlabel"])
    else:
        ax.set_xlabel("Magnetic Field (Oe)")
    if options["ylabel"]:
        ax.set_ylabel(options["ylabel"])
    elif form.lower() == "cdf":
        ax.set_ylabel("Magnetization")
    elif form.lower() == "pdf":
        ax.set_ylabel("dm/dH")
    else:
        raise ValueError(f"`form` argument must be 'pdf' or 'cdf', not {form}")

    _set_ax_lims(ax, form, x, y, max_h_c)

    handle_options(ax, options)

    force_aspect(ax)
    if options["save"]:
        plt.savefig(
            options["save"], dpi=300, bbox_inches="tight", facecolor="w", edgecolor="w"
        )
    ax.legend()
    return fig, ax


def _set_ax_lims(
    ax: plt.Axes,
    plot_type: str,
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    max_h_c: float | None,
) -> None:
    x = np.array(x)
    y = np.array(y)
    if max_h_c:
        if (np.max(x) - np.min(x)) > 50:
            # data is in Oe
            window = 5000 * round(max_h_c / 5000) + 10000
            ax.set_xlim(-window, window)
        else:
            # data is in T
            window = 0.5 * round(max_h_c / 0.5) + 1
            ax.set_xlim(-window, window)

    if plot_type == "cdf":
        m_data_max = max(abs(np.min(y)), abs(np.max(y)))
        ax.set_ylim(-m_data_max * 1.1, m_data_max * 1.1)
    elif plot_type == "pdf":
        ax.set_ylim(-0.1 * np.max(y[10:-10]), 1.1 * np.max(y[10:-10]))
