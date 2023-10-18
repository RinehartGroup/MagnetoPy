from typing import Literal

import matplotlib.pyplot as plt


def handle_kwargs(**kwargs):
    options = {
        "figsize": (5, 5),
        "xlim": None,
        "ylim": None,
        "loc": None,
        "save": None,
        "xlabel": None,
        "ylabel": None,
        "title": None,
    }
    options.update(kwargs)
    return options


def handle_options(
    ax: plt.Axes,
    options: dict[str, str],
    label: str | None = None,
    title: str | None = None,
) -> None:
    if label or title:
        if options["loc"]:
            ax.legend(frameon=False, loc=options["loc"], title=title)
        else:
            ax.legend(frameon=False, loc="best", title=title)
    if options["xlim"]:
        ax.set_xlim(options["xlim"])
    if options["ylim"]:
        ax.set_ylim(options["ylim"])
    if options["xlabel"]:
        ax.set_xlabel(options["xlabel"])
    if options["ylabel"]:
        ax.set_ylabel(options["ylabel"])
    if options["title"]:
        ax.set_title(options["title"])


def get_ylabel(y_val: Literal["moment", "chi", "chi_t"], scaling: list[str]) -> str:
    ylabel = ""
    if y_val == "moment":
        units = "emu"
        if "mass" in scaling:
            units = r"emu g$^{-1}$"
        elif "molar" in scaling:
            units = r"$N_A \cdot \mu_B$"
        ylabel = f"Magnetization ({units})"
    elif y_val == "chi":
        units = r"cm$^3$"
        if "mass" in scaling:
            units = r"cm$^3$ g$^{-1}$"
        elif "molar" in scaling:
            units = r"cm$^3$ mol$^{-1}$"
        ylabel = rf"$\chi$ ({units})"
    elif y_val == "chi_t":
        units = r"cm$^3$ K"
        if "mass" in scaling:
            units = r"cm$^3$ K g$^{-1}$"
        elif "molar" in scaling:
            units = r"cm$^3$ K mol$^{-1}$"
        ylabel = rf"$\chi\cdot$T ({units})"
    return ylabel
