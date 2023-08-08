from typing import Literal


def handle_kwargs(**kwargs):
    options = {"xlim": None, "ylim": None, "loc": None, "save": None}
    options.update(kwargs)
    return options


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


def handle_options(ax, label: str | None, title: str, options: dict[str, str]):
    if label or title:
        if options["loc"]:
            ax.legend(frameon=False, loc=options["loc"], title=title)
        else:
            ax.legend(frameon=False, loc="best", title=title)
    if options["xlim"]:
        ax.set_xlim(options["xlim"])
    if options["ylim"]:
        ax.set_ylim(options["ylim"])