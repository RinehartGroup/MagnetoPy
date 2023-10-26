from typing import Literal


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
