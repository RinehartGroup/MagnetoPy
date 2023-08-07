from typing import Literal
import matplotlib.pyplot as plt

from magnetopy.experiments import MvsH, ZFCFC, ZFC, FC
from magnetopy.plot_helpers import force_aspect, linear_color_gradient, default_colors


def plot_mvsh(
    mvsh: MvsH | list[MvsH],
    normalized: bool = False,
    sequence: str = "",
    colors: str | list[str] = "auto",
    labels: str | list[str] | None = "auto",
    title: str = "",
    **kwargs,
):
    if isinstance(mvsh, list) and len(mvsh) == 1:
        mvsh = mvsh[0]
    if isinstance(mvsh, MvsH):
        if isinstance(colors, list) or isinstance(labels, list):
            raise ValueError(
                "If plotting a single MvsH, `colors` and `labels` must be a single value"
            )
        return plot_single_mvsh(
            mvsh=mvsh,
            normalized=normalized,
            sequence=sequence,
            color=colors,
            label=labels,
            title=title,
            **kwargs,
        )
    if colors != "auto" and not isinstance(colors, list):
        raise ValueError(
            "If plotting multiple MvsH, `colors` must be a list or 'auto'."
        )
    if labels is not None and labels != "auto" and not isinstance(labels, list):
        raise ValueError(
            "If plotting multiple MvsH, `labels` must be a list or 'auto' or `None`."
        )
    return plot_multiple_mvsh(
        mvsh,
        normalized=normalized,
        sequence=sequence,
        colors=colors,
        labels=labels,
        title=title,
        **kwargs,
    )


def plot_single_mvsh(
    mvsh: MvsH,
    normalized: bool = False,
    sequence: str = "",
    color: str = "black",
    label: str | None = "auto",
    title: str = "",
    **kwargs,
):
    options = _handle_kwargs(**kwargs)

    fig, ax = plt.subplots()
    x = mvsh.simplified_data(sequence)["field"] / 10000
    y = mvsh.simplified_data(sequence)["moment"]
    y = y / y.max() if normalized else y
    if label is None:
        ax.plot(x, y, c=color)
    else:
        if label == "auto":
            label = f"{mvsh.temperature} K"
        ax.plot(x, y, c=color, label=label)

    ax.set_xlabel("Field (T)")
    if normalized:
        ax.set_ylabel("Normalized Magnetization")
    else:
        ylabel = _get_ylabel("moment", mvsh.scaling)
        ax.set_ylabel(ylabel)

    _handle_options(ax, label, title, options)

    force_aspect(ax)
    if options["save"]:
        plt.savefig(
            options["save"], dpi=300, bbox_inches="tight", facecolor="w", edgecolor="w"
        )
    return fig, ax


def plot_multiple_mvsh(
    mvsh: list[MvsH],
    normalized: bool = False,
    sequence: str = "",
    colors: list[str] | Literal["auto"] = "auto",
    labels: list[str] | None = None,
    title: str = "",
    **kwargs,
):
    options = _handle_kwargs(**kwargs)

    if colors == "auto":
        colors = default_colors(len(mvsh))
    if _check_if_variable_temperature(mvsh):
        mvsh.sort(key=lambda x: x.temperature)
        colors = linear_color_gradient("blue", "red", len(mvsh))
        if labels == "auto":
            labels = [f"{x.temperature} K" for x in mvsh]
    if labels is None:
        labels: list[None] = [None] * len(mvsh)

    fig, ax = plt.subplots()
    for m, color, label in zip(mvsh, colors, labels):
        x = m.simplified_data(sequence)["field"] / 10000
        y = m.simplified_data(sequence)["moment"]
        y = y / y.max() if normalized else y
        if label:
            ax.plot(x, y, c=color, label=label)
        else:
            ax.plot(x, y, c=color)

    ax.set_xlabel("Field (T)")
    if normalized:
        ax.set_ylabel("Normalized Magnetization")
    else:
        ylabel = _get_ylabel("moment", mvsh[0].scaling)
        ax.set_ylabel(ylabel)

    _handle_options(ax, labels[0], title, options)
    force_aspect(ax)
    if options["save"]:
        plt.savefig(
            options["save"], dpi=300, bbox_inches="tight", facecolor="w", edgecolor="w"
        )
    return fig, ax


def plot_zfcfc(
    zfc: ZFC | list[ZFC],
    fc: FC | list[FC],
    y_val: Literal["moment", "chi", "chi_t"] = "moment",
    normalized: bool = False,
    colors: str | list[str] = "auto",
    labels: str | list[str] | None = "auto",
    title: str = "",
    **kwargs,
):
    if isinstance(zfc, list) and len(zfc) == 1:
        zfc = zfc[0]
    if isinstance(fc, list) and len(fc) == 1:
        fc = fc[0]
    if isinstance(zfc, ZFC) and isinstance(fc, FC):
        if isinstance(colors, list) or isinstance(labels, list):
            raise ValueError(
                "If plotting a single ZFCFC, `colors` and `labels` must be a single value"
            )
        return plot_single_zfcfc(
            zfc=zfc,
            fc=fc,
            y_val=y_val,
            normalized=normalized,
            color="black" if colors == "auto" else colors,
            label=labels,
            title=title,
            **kwargs,
        )
    if not isinstance(zfc, list) or not isinstance(fc, list) or (len(zfc) != len(fc)):
        raise ValueError("ZFC and FC must be the same length")
    if colors != "auto" and not isinstance(colors, list):
        raise ValueError(
            "If plotting multiple ZFCFC, `colors` must be a list or 'auto'."
        )
    if labels is not None and labels != "auto" and not isinstance(labels, list):
        raise ValueError(
            "If plotting multiple ZFCFC, `labels` must be a list or 'auto' or `None`."
        )
    zfc.sort(key=lambda x: x.field)
    fc.sort(key=lambda x: x.field)
    for zfc_i, fc_i in zip(zfc, fc):
        if zfc_i.field != fc_i.field:
            raise ValueError("ZFC and FC must have the same fields")
    return plot_multiple_zfcfc(
        zfc,
        fc,
        y_val=y_val,
        normalized=normalized,
        colors=colors,
        labels=labels,
        title=title,
        **kwargs,
    )


def plot_single_zfcfc(
    zfc: ZFC,
    fc: FC,
    y_val: Literal["moment", "chi", "chi_t"] = "moment",
    normalized: bool = False,
    color: str = "black",
    label: str | None = "auto",
    title: str = "",
    **kwargs,
):
    options = _handle_kwargs(**kwargs)

    fig, ax = plt.subplots()
    x_zfc = zfc.simplified_data()["temperature"]
    y_zfc = zfc.simplified_data()[y_val]
    y_zfc = y_zfc / y_zfc.max() if normalized else y_zfc
    x_fc = fc.simplified_data()["temperature"]
    y_fc = fc.simplified_data()[y_val]
    y_fc = y_fc / y_fc.max() if normalized else y_fc
    if label is None:
        ax.plot(x_zfc, y_zfc, c=color, ls="--")
        ax.plot(x_fc, y_fc, c=color)
    else:
        if label == "auto":
            if zfc.field > 10000:
                label = f"{(zfc.field / 10000):.0f} T"
            else:
                label = f"{zfc.field:.0f} Oe"
        ax.plot(x_zfc, y_zfc, c=color, ls="--", label=label)
        ax.plot(x_fc, y_fc, c=color, label="")

    ax.set_xlabel("Temperature (K)")
    if normalized:
        if y_val == "moment":
            normalized_ylabel = "Normalized Magnetization"
        elif y_val == "chi":
            normalized_ylabel = r"Normalized $\chi$"
        else:
            normalized_ylabel = r"Normalized $\chi\cdot$T"
        ax.set_ylabel(normalized_ylabel)
    else:
        ylabel = _get_ylabel(y_val, zfc.scaling)
        ax.set_ylabel(ylabel)

    _handle_options(ax, label, title, options)

    force_aspect(ax)
    if options["save"]:
        plt.savefig(
            options["save"], dpi=300, bbox_inches="tight", facecolor="w", edgecolor="w"
        )
    return fig, ax


def plot_multiple_zfcfc(
    zfc: list[ZFC],
    fc: list[FC],
    y_val: Literal["moment", "chi", "chi_t"] = "moment",
    normalized: bool = False,
    colors: list[str] | Literal["auto"] = "auto",
    labels: list[str] | None = None,
    title: str = "",
    **kwargs,
):
    options = _handle_kwargs(**kwargs)

    if colors == "auto":
        colors = default_colors(len(zfc))
    if _check_if_variable_field(zfc):
        zfc.sort(key=lambda x: x.field)
        colors = linear_color_gradient("purple", "green", len(zfc))
        if labels == "auto":
            labels = [f"{x.field:.0f} Oe" for x in zfc]
    if labels is None:
        labels: list[None] = [None] * len(zfc)

    fig, ax = plt.subplots()
    for zfc_i, fc_i, color, label in zip(zfc, fc, colors, labels):
        x_zfc = zfc_i.simplified_data()["temperature"]
        y_zfc = zfc_i.simplified_data()[y_val]
        y_zfc = y_zfc / y_zfc.max() if normalized else y_zfc
        x_fc = fc_i.simplified_data()["temperature"]
        y_fc = fc_i.simplified_data()[y_val]
        y_fc = y_fc / y_fc.max() if normalized else y_fc
        if label:
            ax.plot(x_zfc, y_zfc, c=color, ls="--", label=label)
            ax.plot(x_fc, y_fc, c=color, label="")
        else:
            ax.plot(x_zfc, y_zfc, c=color, ls="--")
            ax.plot(x_fc, y_fc, c=color)

    ax.set_xlabel("Temperature (K)")
    if normalized:
        if y_val == "moment":
            normalized_ylabel = "Normalized Magnetization"
        elif y_val == "chi":
            normalized_ylabel = r"Normalized $\chi$"
        else:
            normalized_ylabel = r"Normalized $\chi\cdot$T"
        ax.set_ylabel(normalized_ylabel)
    else:
        ylabel = _get_ylabel(y_val, zfc[0].scaling)
        ax.set_ylabel(ylabel)

    _handle_options(ax, labels[0], title, options)

    force_aspect(ax)
    if options["save"]:
        plt.savefig(
            options["save"], dpi=300, bbox_inches="tight", facecolor="w", edgecolor="w"
        )
    return fig, ax


def _handle_kwargs(**kwargs):
    options = {"xlim": None, "ylim": None, "loc": None, "save": None}
    options.update(kwargs)
    return options


def _get_ylabel(y_val: Literal["moment", "chi", "chi_t"], scaling: list[str]) -> str:
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


def _handle_options(ax, label: str | None, title: str, options: dict[str, str]):
    if label or title:
        if options["loc"]:
            ax.legend(frameon=False, loc=options["loc"], title=title)
        else:
            ax.legend(frameon=False, loc="best", title=title)
    if options["xlim"]:
        ax.set_xlim(options["xlim"])
    if options["ylim"]:
        ax.set_ylim(options["ylim"])


def _check_if_variable_temperature(mvsh: list[MvsH]):
    first_temp = mvsh[0].temperature
    for mvsh_obj in mvsh:
        if mvsh_obj.temperature != first_temp:
            return True
    return False


def _check_if_variable_field(zfc: list[ZFC]):
    first_field = zfc[0].field
    for zfc_obj in zfc:
        if zfc_obj.field != first_field:
            return True
    return False


# def plot_voltage_scan(
#     scan_obj: SingleRawDCScan,
#     yaxis: str = "free_c",
# ):

#     scan = scan_obj.QD_fit_scan
#     fig, ax = plt.subplots()
#     ax.set_xlabel("Position (mm)")
#     ylabel = {"fixed_c": "Fixed C Fitted (V)", "free_c": "Free C Fitted (V)"}[yaxis]
#     ax.set_ylabel(ylabel)
#     x = scan["Raw Position (mm)"]
#     y = scan[ylabel]
#     ax.plot(x, y)
#     return fig, ax


# def plot_analyzed_voltage_scan(analyzed_scan_obj: AnalyzedSingleRawDCScan, mult=10):
#     df = analyzed_scan_obj.scan
#     fig, ax = plt.subplots()

#     ax.set_xlabel("Position (mm)")
#     ax.set_ylabel("Voltage (V)")
#     x = df["Raw Position (mm)"]
#     y1 = df["Free C Fitted (V)"]
#     y2 = df["Fit (V)"]
#     y3 = df["Residual"]
#     ax.scatter(x, y1, s=5, c="blue", label="Data")
#     ax.plot(x, y2, c="red", label="Fit")
#     ax.plot(x, y3 * mult, c="black", label=f"Residual (x{mult})")
#     ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
#     force_aspect(ax)
#     return fig, ax


# def plot_all_in_single_voltage_scan(scan_obj: SingleRawDCScan):
#     fig, ax = plt.subplots()
#     x1 = scan_obj.up_scan["Raw Position (mm)"]
#     x2 = scan_obj.down_scan["Raw Position (mm)"]
#     x3 = scan_obj.scan["Raw Position (mm)"]
#     ax.plot(x1, scan_obj.up_scan["Raw Voltage (V)"], c="red", ls="--", label="up-raw")
#     ax.plot(
#         x1,
#         scan_obj.up_scan["Processed Voltage (V)"],
#         c="black",
#         ls="--",
#         label="up-processed",
#     )
#     ax.plot(
#         x2, scan_obj.down_scan["Raw Voltage (V)"], c="red", ls="-", label="down-raw"
#     )
#     ax.plot(
#         x2,
#         scan_obj.down_scan["Processed Voltage (V)"],
#         c="black",
#         ls="-",
#         label="down-processed",
#     )
#     ax.plot(x3, scan_obj.scan["Free C Fitted (V)"], c="blue", label="avg_free_c")
#     ax.plot(x3, scan_obj.scan["Fixed C Fitted (V)"], c="green", label="avg_fixed_c")
#     force_aspect(ax)
#     ax.set_xlabel("Position (mm)")
#     ax.set_ylabel("Voltage (V)")
#     ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
#     return fig, ax


# def plot_analyses(
#     df_list: list,
#     y_val: str = "Residuals",
#     colors: list | None = None,
#     labels: list | None = None,
#     title: str = "",
# ):
#     """
#     Plots analyses from multiple scans
#     y_val options are 'Fit' and 'Residual'
#     df_list must contain columns ['Position', 'Voltage', 'Fit', 'Residual']
#     """
#     n = len(df_list)
#     if colors is None:
#         colors = linear_color_gradient("purple", "orange", n)
#     show_label = True
#     if labels is None:
#         show_label = False
#         labels = [""] * n
#     fig, ax = plt.subplots()
#     ax.set_xlabel("Position (mm)")
#     ax.set_ylabel("Resdidual (V)")
#     if y_val == "Residual":
#         ax.set_ylabel("Resdidual (V)")
#         for df, color, label in zip(df_list, colors, labels):
#             ax.plot(
#                 df["Position"],
#                 df["Residual"],
#                 c=color,
#                 label=label,
#             )
#     if y_val == "Fit":
#         ax.set_ylabel("Voltage")
#         for df, color, label in zip(df_list, colors, labels):
#             ax.plot(df["Position"], df["Voltage"], c=color, label=label, ls="-")
#             ax.plot(df["Position"], df["Fit"], c=color, label="", ls="--")
#     if show_label:
#         ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", title=title)
#     force_aspect(ax)
#     return fig, ax


# def plot_zfcfc(
#     pairs: Tuple[QDFile, QDFile] | list[Tuple[QDFile, QDFile]],
#     y_val: str = "DC Moment Free Ctr (emu)",
#     normalized=False,
#     colors: list | None = None,
#     labels: list | None = None,
#     title: str = "",
#     **kwargs,
# ):
#     """
#     Plot ZFC / FC curves for either a single dataset or multiple
#     The tuples in 'pairs' should be listed as (ZFC, FC)
#     """
#     options = {"xlim": None, "ylim": None, "loc": None, "save": None}
#     options.update(kwargs)
#     if isinstance(pairs, list) and colors is None:
#         colors = linear_color_gradient("purple", "orange", len(pairs))
#     show_label = True
#     if isinstance(pairs, tuple):
#         pairs = [pairs]
#         colors = ["black"]
#         labels = [labels]
#         show_label = False
#     fig, ax = plt.subplots()
#     ax.set_xlabel("Temperature (K)")

#     ylabel_dict = {
#         "DC Moment Free Ctr (emu)": "Magnetization (emu)",
#         "Moment_per_mass": "Magnetization (emu/g)",
#     }
#     ylabel = ylabel_dict[y_val]
#     if normalized:
#         ylabel = "Normalized Magnetization (emu)"
#     ax.set_ylabel(ylabel)

#     for pair, color, label in zip(pairs, colors, labels):
#         x_zfc = pair[0].parsed_data["Temperature (K)"]
#         y_zfc = pair[0].parsed_data[y_val]
#         x_fc = pair[1].parsed_data["Temperature (K)"]
#         y_fc = pair[1].parsed_data[y_val]
#         y_zfc = y_zfc / y_fc.max() if normalized else y_zfc
#         y_fc = y_fc / y_fc.max() if normalized else y_fc
#         ax.plot(x_zfc, y_zfc, c=color, label="", ls="--")
#         ax.plot(x_fc, y_fc, c=color, label=label, ls="-")
#     if options["xlim"]:
#         ax.set_xlim(options["xlim"])
#     if options["ylim"]:
#         ax.set_ylim(options["ylim"])
#     if show_label or title is not None or labels is not None:
#         if options["loc"]:
#             ax.legend(loc=options["loc"], title=title)
#         else:
#             ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", title=title)
#     force_aspect(ax)
#     if options["save"]:
#         plt.savefig(
#             options["save"], dpi=300, bbox_inches="tight", facecolor="w", edgecolor="w"
#         )
#     return fig, ax


# def plot_zfcfc_w_blocking(
#     pairs: Tuple[QDFile, QDFile] | list[Tuple[QDFile, QDFile]],
#     y_val: str = "DC Moment Free Ctr (emu)",
#     normalized=False,
#     colors: list | None = None,
#     labels: list | None = None,
#     title: str = "",
#     **kwargs,
# ):
#     """
#     Plot ZFC / FC curves for either a single dataset or multiple
#     The tuples in 'pairs' should be listed as (ZFC, FC)
#     """
#     options = {"xlim": None, "ylim": None, "loc": None, "figsize": None, "save": None}
#     options.update(kwargs)
#     if isinstance(pairs, list) and colors is None:
#         colors = linear_color_gradient("purple", "orange", len(pairs))
#     show_label = True
#     if isinstance(pairs, tuple):
#         pairs = [pairs]
#         colors = ["black"]
#         labels = [labels]
#         show_label = False

#     figsize = (7, 7) if options["figsize"] is None else options["figsize"]
#     fig, axs = plt.subplots(2, 1, sharex=True, figsize=figsize)
#     fig.subplots_adjust(hspace=0)
#     axs[1].set_xlabel("Temperature (K)")

#     ylabel_dict = {
#         "DC Moment Free Ctr (emu)": "Magnetization (emu)",
#         "Moment_per_mass": "Magnetization (emu/g)",
#     }
#     ylabel = ylabel_dict[y_val]
#     if normalized:
#         ylabel = "Normalized Magnetization (emu)"
#     axs[0].set_ylabel(ylabel)
#     axs[1].set_ylabel("$d(M_{FC} - M_{ZFC} / dT$ (emu/K)")
#     axs[1].set_yticklabels([])

#     for pair, color, label in zip(pairs, colors, labels):
#         x_zfc = pair[0].parsed_data["Temperature (K)"]
#         y_zfc = pair[0].parsed_data[y_val]
#         x_fc = pair[1].parsed_data["Temperature (K)"]
#         y_fc = pair[1].parsed_data[y_val]
#         y_zfc = y_zfc / y_fc.max() if normalized else y_zfc
#         y_fc = y_fc / y_fc.max() if normalized else y_fc
#         axs[0].plot(x_zfc, y_zfc, c=color, label="", ls="--")
#         axs[0].plot(x_fc, y_fc, c=color, label=label, ls="-")
#         tb_temp, tb_df = determine_blocking_temp(pair[0], pair[1])
#         axs[1].plot(tb_df["Temperature (K)"], tb_df["deriv"], c=color, label="")
#     if options["xlim"]:
#         axs[1].set_xlim(options["xlim"])
#     if options["ylim"]:
#         axs[0].set_ylim(options["ylim"])
#     if show_label or title is not None or labels is not None:
#         if options["loc"]:
#             axs[0].legend(loc=options["loc"], title=title)
#         else:
#             axs[0].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", title=title)
#     force_aspect(axs[0])
#     force_aspect(axs[1])
#     if options["save"]:
#         plt.savefig(
#             options["save"], dpi=300, bbox_inches="tight", facecolor="w", edgecolor="w"
#         )
#     return fig, axs
