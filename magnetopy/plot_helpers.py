from typing import Union
import numpy as np
import pandas as pd


def linear_color_gradient(
    start_hex: str,
    finish_hex: str = "#FFFFFF",
    n: Union[int, pd.core.series.Series] = 10,
):
    """
    Modified from https://bsouthga.dev/posts/color-gradients-with-python
    returns a gradient list of (n) colors between two hex colors.
    start_hex and finish_hex should be the full six-digit color string, inlcuding the '#' sign ("#FFFFFF")
    """

    def hex_to_RGB(hex: str) -> list:
        # Pass 16 to the integer function for change of base
        return [int(hex[i : i + 2], 16) for i in range(1, 6, 2)]

    def RGB_to_hex(RGB: str) -> str:
        # Components need to be integers for hex to make sense
        RGB = [int(x) for x in RGB]
        return "#" + "".join(
            ["0{0:x}".format(v) if v < 16 else "{0:x}".format(v) for v in RGB]
        )

    default_colors = {
        "red": "#FF0000",
        "orange": "#E36D12",
        "yellow": "#FFFF00",
        "green": "#00FF00",
        "blue": "#0000FF",
        "purple": "#6E12E6",
        "black": "#000000",
        "white": "#FFFFFF",
    }
    start_hex = default_colors.get(start_hex, start_hex)
    finish_hex = default_colors.get(finish_hex, finish_hex)

    s = hex_to_RGB(start_hex)
    f = hex_to_RGB(finish_hex)
    # Initilize a list of the output colors with the starting color
    RGB_list = [s]
    # Calcuate a color at each evenly spaced value of t from 1 to n
    if isinstance(n, int):
        for t in range(1, n):
            # Interpolate RGB vector for color at the current value of t
            curr_vector = [
                int(s[j] + (float(t) / (n - 1)) * (f[j] - s[j])) for j in range(3)
            ]
            # Add it to our list of output colors
            RGB_list.append(curr_vector)
    elif isinstance(n, pd.core.series.Series):
        full_vector = [f[i] - s[i] for i in range(3)]
        full_vector_length = np.sqrt(
            full_vector[0] ** 2 + full_vector[1] ** 2 + full_vector[2] ** 2
        )
        unit_vector = [full_vector[i] / full_vector_length for i in range(3)]
        index_range = n.max() - n.min()
        previous_color = s
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
            RGB_list.append(curr_vector)

    hex_list = []
    for RGB in RGB_list:
        hex_list.append(RGB_to_hex(RGB))

    return hex_list


def force_aspect(ax, aspect=1):
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
