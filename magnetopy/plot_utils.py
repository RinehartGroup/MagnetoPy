from typing import Sized, TypeAlias, Literal, NewType
import numpy as np
import pandas as pd
from itertools import cycle

import matplotlib.pyplot as plt

BasicColors: TypeAlias = Literal[
    "red", "orange", "yellow", "green", "blue", "purple", "black", "white"
]
HexColorCode = NewType("HexColorCode", str)


def linear_color_gradient(
    start_color: HexColorCode | BasicColors,
    finish_color: HexColorCode | BasicColors = "white",
    n: int | Sized = 10,
) -> list[HexColorCode]:
    """Return a list of colors forming linear gradients between two colors.

    Parameters
    ----------
    start_color : HexColorCode | basic_colors
        The starting color as either a hex code (e.g. "#FFFFFF") or a basic color
        name.
    finish_color : HexColorCode | basic_colors, optional
        The ending color as either a hex code (e.g. "#FFFFFF") or a basic color
        name, by default "white"
    n : int | Sized, optional
        The number of colors to return, by default 10.

    Returns
    -------
    list[HexColorCode]
        A list of hex color codes forming a linear gradient between the start and
        finish colors.

    Notes
    -----
    Supported basic colors includes: "red", "orange", "yellow", "green", "blue",
    "purple", "black", and "white".

    References
    ----------
    [Color Gradients with Python](
        https://bsouthga.dev/posts/color-gradients-with-python)
    """

    def hex_to_rgb(hex_color: str) -> list:
        # Pass 16 to the integer function for change of base
        return [int(hex_color[i : i + 2], 16) for i in range(1, 6, 2)]

    def rgb_to_hex(rgb: str) -> str:
        # Components need to be integers for hex to make sense
        rgb = [int(x) for x in rgb]
        return "#" + "".join(
            ["0{0:x}".format(v) if v < 16 else "{0:x}".format(v) for v in rgb]
        )

    _basic_colors = {
        "red": "#FF0000",
        "orange": "#E36D12",
        "yellow": "#FFFF00",
        "green": "#0F7823",
        "blue": "#0000FF",
        "purple": "#6E12E6",
        "black": "#000000",
        "white": "#FFFFFF",
    }
    start_color = _basic_colors.get(start_color, start_color)
    finish_color = _basic_colors.get(finish_color, finish_color)

    start = hex_to_rgb(start_color)
    finish = hex_to_rgb(finish_color)
    # Initilize a list of the output colors with the starting color
    rgb_list = [start]
    # Calcuate a color at each evenly spaced value of t from 1 to n
    if isinstance(n, int):
        for t in range(1, n):
            # Interpolate RGB vector for color at the current value of t
            curr_vector = [
                int(start[j] + (float(t) / (n - 1)) * (finish[j] - start[j]))
                for j in range(3)
            ]
            # Add it to our list of output colors
            rgb_list.append(curr_vector)
    elif isinstance(n, pd.core.series.Series):
        full_vector = [finish[i] - start[i] for i in range(3)]
        full_vector_length = np.sqrt(
            full_vector[0] ** 2 + full_vector[1] ** 2 + full_vector[2] ** 2
        )
        unit_vector = [full_vector[i] / full_vector_length for i in range(3)]
        index_range = n.max() - n.min()
        previous_color = start
        n = n.to_list()
        previous_index = n[0]
        for current_index in n[1:]:
            step_size = (
                full_vector_length * (current_index - previous_index) / index_range
            )
            # Make step of size step_size in direction of full_vector
            curr_vector = [
                previous_color[i] + unit_vector[i] * step_size for i in range(3)
            ]
            previous_index = current_index
            previous_color = curr_vector
            rgb_list.append(curr_vector)

    hex_list = []
    for rgb in rgb_list:
        hex_list.append(rgb_to_hex(rgb))

    return hex_list


def default_colors(n: int | Sized) -> list[str]:
    """Return a list of colors for plotting. The default colors are the first 10
    colors from the Matplotlib default color cycle. If more than 10 colors are
    requested, the default colors are cycled.

    Parameters
    ----------
    n : int | Sized
        The number of colors to return or a list-like object of the same length as
        the number of colors to return.

    Returns
    -------
    list[str]
        A list of colors for plotting.
    """
    if not isinstance(n, int):
        n = len(n)
    default = cycle(
        [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
    )
    return [next(default) for _ in range(n)]


def force_aspect(ax: plt.Axes, aspect=1) -> None:
    """Force the aspect ratio of a plot to be a certain value. Uses the axes's current
    x and y limits to calculate the aspect ratio. Works for both linear and log scales.

    Parameters
    ----------
    ax : plt.Axes
        The axes to force the aspect ratio of.
    aspect : int, optional
        The desired aspect ratio, by default 1.
    """
    # aspect is width/height
    xscale_str = ax.get_xaxis().get_scale()
    yscale_str = ax.get_yaxis().get_scale()
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    if xscale_str == "linear" and yscale_str == "linear":
        asp = abs((xmax - xmin) / (ymax - ymin)) / aspect
    elif xscale_str == "log" and yscale_str == "linear":
        asp = abs((np.log10(xmax) - np.log10(xmin)) / (ymax - ymin)) / aspect
    elif xscale_str == "log" and yscale_str == "log":
        asp = (
            abs((np.log10(xmax) - np.log10(xmin)) / (np.log10(ymax) - np.log10(ymin)))
            / aspect
        )
    elif xscale_str == "linear" and yscale_str == "log":
        asp = abs((xmax - xmin) / (np.log10(ymax) - np.log10(ymin))) / aspect
    ax.set_aspect(asp)
