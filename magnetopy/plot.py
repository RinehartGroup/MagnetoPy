# import pathlib
# from typing import Tuple

# import matplotlib as mpl
# import matplotlib.pyplot as plt

# from magnetopy.fits import arctan_fit, determine_blocking_temp
# from magnetopy.parse_qd import AnalyzedSingleRawDCScan, QDFile, SingleRawDCScan
# from magnetopy.plot_helpers import force_aspect, linear_color_gradient


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


# def plot_mvsh(
#     sweeps: QDFile | list[QDFile],
#     y_val: str = "DC Moment Free Ctr (emu)",
#     normalized=False,
#     colors: list | str | None = None,
#     labels: list | str | None = "",
#     title=None,
#     **kwargs,
# ):
#     """
#     y_val options:
#         'DC Moment Free Ctr (emu)'
#         'Background Subtracted Moment (emu)'
#     """
#     options = {"xlim": None, "ylim": None, "loc": None, "save": None}
#     options.update(kwargs)
#     if isinstance(sweeps, list) and colors is None:
#         colors = linear_color_gradient("purple", "orange", len(sweeps))
#     show_label = True
#     if isinstance(sweeps, QDFile):
#         sweeps = [sweeps]
#         colors = ["black"]
#         labels = [labels]
#         show_label = False
#     fig, ax = plt.subplots()
#     ax.set_xlabel("Field (T)")

#     ylabel_dict = {
#         "DC Moment Free Ctr (emu)": "Magnetization (emu)",
#         "Moment_per_mass": "Magnetization (emu/g)",
#     }
#     ylabel = ylabel_dict[y_val]
#     if normalized:
#         ylabel = "Normalized Magnetization (emu)"
#     ax.set_ylabel(ylabel)

#     for sweep, color, label in zip(sweeps, colors, labels):
#         x = sweep.parsed_data["Magnetic Field (Oe)"] / 10000
#         y = sweep.parsed_data[y_val]
#         y = y / y.max() if normalized else y
#         ax.plot(x, y, c=color, label=label)
#     if show_label or title is not None or labels is not None:
#         if options["loc"]:
#             ax.legend(loc=options["loc"], title=title)
#         else:
#             ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", title=title)
#     if options["xlim"]:
#         ax.set_xlim(options["xlim"])
#     if options["ylim"]:
#         ax.set_ylim(options["ylim"])
#     force_aspect(ax)
#     if options["save"]:
#         plt.savefig(
#             options["save"], dpi=300, bbox_inches="tight", facecolor="w", edgecolor="w"
#         )
#     return fig, ax


# def plot_mvsh_w_fits(
#     sweeps: QDFile | list[QDFile],
#     y_val: str = "DC Moment Free Ctr (emu)",
#     normalized=False,
#     colors: list | None = None,
#     labels: list | None = None,
#     title=None,
#     **kwargs,
# ):
#     """
#     y_val options:
#         'DC Moment Free Ctr (emu)'
#         'Background Subtracted Moment (emu)'
#     """
#     options = {"xlim": None, "ylim": None, "loc": None, "figsize": None, "save": None}
#     options.update(kwargs)
#     if isinstance(sweeps, list) and colors is None:
#         colors = linear_color_gradient("purple", "orange", len(sweeps))
#     show_label = True
#     if isinstance(sweeps, QDFile):
#         sweeps = [sweeps]
#         colors = ["black"]
#         labels = [labels]
#         show_label = False
#     fig, ax = plt.subplots()
#     ax.set_xlabel("Field (T)")

#     ylabel_dict = {
#         "DC Moment Free Ctr (emu)": "Magnetization (emu)",
#         "Moment_per_mass": "Magnetization (emu/g)",
#     }
#     ylabel = ylabel_dict[y_val]
#     if normalized:
#         ylabel = "Normalized Magnetization (emu)"
#     moment_per_mass = True if y_val == "Moment_per_mass" else False
#     ax.set_ylabel(ylabel)

#     for sweep, color, label in zip(sweeps, colors, labels):
#         x = sweep.parsed_data["Magnetic Field (Oe)"] / 10000
#         y = sweep.parsed_data[y_val]
#         y = y / y.max() if normalized else y
#         ax.scatter(x, y, c=color, s=5, label=label)
#         popt, fit_df = arctan_fit(
#             sweep, normalized=normalized, moment_per_mass=moment_per_mass
#         )
#         x_fit = fit_df["field"] / 10000
#         y_fit = fit_df["moment"]
#         ax.plot(x_fit, y_fit, c=color)
#     if show_label or title is not None or labels is not None:
#         if options["loc"]:
#             ax.legend(loc=options["loc"], title=title)
#         else:
#             ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", title=title)
#     if options["xlim"]:
#         ax.set_xlim(options["xlim"])
#     if options["ylim"]:
#         ax.set_ylim(options["ylim"])
#     force_aspect(ax)
#     if options["save"]:
#         plt.savefig(
#             options["save"], dpi=300, bbox_inches="tight", facecolor="w", edgecolor="w"
#         )
#     return fig, ax
