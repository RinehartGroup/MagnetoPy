import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from magnetopy.analyses.cauchy.standalone import (
    CauchyFittingArgs,
    CauchyAnalysisResults,
)
from magnetopy.experiments.plot_utils import (
    handle_kwargs,
    handle_options,
)
from magnetopy.plot_utils import force_aspect


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
        Assuming results is not None, whether to show the full fit, by default True.
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
    ax.plot(x, y, label="Data", color="black")
    if input_params:
        x_input = np.linspace(min(x), max(x), 1000)
        y_input = input_params.generate_pdf_data(x_input)
        ax.plot(x_input, y_input, label="Input", color="blue")
        max_h_c = np.max([term.h_c for term in input_params.terms])
    if results:
        x_fit = np.linspace(min(x), max(x), 1000)
        y_fit = results.simulate_pdf_data(x_fit)
        max_h_c = np.max([term.h_c for term in results.terms])
        if show_full_fit:
            ax.plot(x_fit, y_fit, label="Fit", color="red")
        if show_fit_components:
            y_fits = results.simulate_pdf_data_by_term(x_fit)
            for i, y_fit in enumerate(y_fits):
                ax.plot(x_fit, y_fit, label=f"Fit Term {i}")

    if options["xlabel"]:
        ax.set_xlabel(options["xlabel"])
    else:
        ax.set_xlabel("Magnetic Field (Oe)")
    if options["ylabel"]:
        ax.set_ylabel(options["ylabel"])
    else:
        ax.set_ylabel(r"$dm/dH$ (emu/Oe)")

    _set_ax_lims(ax, "pdf", x, y, max_h_c)

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
        m_data_max = max(abs(np.min(y), abs(np.max(y))))
        ax.set_ylim(-m_data_max * 1.1, m_data_max * 1.1)
    elif plot_type == "pdf":
        ax.set_ylim(-0.1 * np.max(y[10:-10]), 1.1 * np.max(y[10:-10]))
