# MagnetoPy

`MagnetoPy` is the magnetic successor to the `magdb` set of packages. At the time, I was working on low-moment magnetic data analysis, with the intent to develop code for doing our own background subtraction, moment determination, and data validation.

Here is a breakdown of the modules, in order of likely usefulness:

- `parse_qd` contains class definitions for:
  - `QDFile`, which you can create by passing a path to a QD file. This handles parsing of .dat and .rw.dat files, where if the file is a .dat file _and_ the .rw.dat equivalent exists, the raw scans are included in the `QDFile` object.
  - `SingleRawDCScan` contains the data and ancillary info found in each voltage vs position scan in the .rw.dat file. It is used in the creation of `QDFile` objects.
  - `AnalyzedRawDCSCan` is meant for raw scan analysis. It will fit a voltage vs position scan and stores the fit values. It's called using `QDFile.analyze_raw()`, but there's a chance this was done incorrectly and it's using the wrong scan (QD files have the up and down scans in addition to a number of processed scans, I may have used the wrong one here, but it was a while ago and the project didn't continue to go that direction).

- `plots` contains functions for plotting. Typically you call them by passing a `QDFile` or other objects above, along with some plotting parameters. The functions are:
  - `plot_zfcfc()`
  - `plot_zfcfc_w_blocking()` - plots zfcfc with a line representing the blocking temperature as determined by `fits.determine_blocking_temperature()`
  - `plot_mvsh()`
  - `plot_mvsh_w_fits()` - includes fits using `fits.arctan_fit()`
  - `plot_voltage_scan()`, `plot_analyzed_voltage_scan()`, `plot_all_in_single_voltage_scan()`, `plot_analyses()` are all related to raw voltage scans

- `plot_helpers` includes two useful functions I use in almost all of my plotting with `matplotlib`:
  - `linear_color_gradients(start_hex: str, finish_hex: str = #FFFFFF, n: Union[int, pd.core.series.Series] = 10)` creates a list of _n_ hex colors that create a smooth gradient between `start_hex` and `finish_hex`.
    - `start_hex` and `finish_hex` are strings are of the form #RRGGBB, where RR, GG, and BB are the red, green, and blue values in hex. Alternatively you can pass simple colors to it, e.g. 'red', 'blue', etc. Default behavior for `finish_hex` is black.
    - `n` can be either a number or an iterable (most likely a pd.Series). Likely you'll use that if you're passing a list of pd.DataFrames or a list of `QDFile` objects, and you want to color each one differently.
  - `force_aspect()` - getting figures with 1:1 aspect ratios in `matplotlib` is a pain because of the way it handles axes. This function forces the aspect ratio to 1:1 and works with linear, partial log, and log-log plots.

- `calibration` contains the `Calibration` class, which was meant to be used as a one time call of `QDFile` using files collected with the palladium standard (stored in `/cal_and_test_data/calibration`). The fit parameters are then used to convert sample fit parameters (I think just the amplitude, _A_) to moments.

  The idea was that for most samples you could use the typical calibration data stored in `/cal_and_test_data/calibration`, then if you are using a special sample holder you would collect data on the palladium standard in that holder and add it to the `/calibration` folder.

- `bkg_analysis` is where the work on background subtraction started to tail off. There are some functions in there to get things started and some empty classes to give you and idea of what I was thinking.

## Raw Data Processing, MagnetoPy, and magdb

A lot of the magnetism code in `magdb` takes from `MagnetoPy`, though not everything has been implemented yet. As of 10/12/2022, mostly what has made it into `magdb` is parsing of .dat and .rw.dat files. `magdb` has more sophisticated labeling of measurement types (zfc, fc, mvsh, etc.) but doesn't have any of the raw data analysis (beyond parsing) and plotting. 

I don't plan on continuing to develop `MagnetoPy`, but it might be a good starting point for continued work on the raw data analysis. The code that Kyle used to do the parsing portion of the raw data plotting is all in `parse_qd`. Anything that gets done here will help start the process of getting it into `magdb`.

## Report Creation

The one thing I still use `MagnetoPy` for is report creation. I have an external SQUID user that runs the same measurements on every sample, so I have a somewhat polishhed script for generating the figures and csv files for the report. I included a sample notebook, [`example_report_creation.ipynb`](https://github.com/RinehartGroup/MagnetoPy/blob/master/examples/report_creation/example_report_creation.ipynb), with more info there on how that works.

If we start doing a bunch of measurements for outside users, we might consider making some standard scripts for report creation. 

## Installation and Usage

> **Recommendation:**
> Use virtual environments to separate the different projects you work on. Definitely use a virtual environment if you want to work on developing the package. Here's some potentially helpful info to get you started:
> - [official `venv` docs:](https://docs.python.org/3/library/venv.html)
> - [a more thorough tutorial on virtual environments](https://realpython.com/python-virtual-environments-a-primer/)
> - [using Jupyter Notebooks / ipython with virtual environments](https://janakiev.com/blog/jupyter-virtual-envs/)


If you just want to use the code, you can install it with `pip install git+https://github.com/RinehartGroup/MagnetoPy.git`. I'm not a `conda` user, [but this should help](https://stackoverflow.com/questions/19042389/conda-installing-upgrading-directly-from-github). The necessary requirements should install automatically, but if they don't check out `requirements.txt` and make sure you have the requisite packages.

Importing `magnetopy` exposes all classes and functions, as I show in [`example_report_creation.ipynb`](https://github.com/RinehartGroup/MagnetoPy/blob/master/examples/report_creation/example_report_creation.ipynb).

### Development

If you want to work on the code, you can:

- fork the repository so you have a copy on your own github account. Go to that repository and clone it to your local machine.
- create a branch on this repository (name it either after yourself or the feature you're working on), clone the repository to your local machine, then __switch to that branch__ using `git checkout <branch name>`.

Once whatever you're working on is stable, make sure it doesn't break any of the existing tests/examples, consider adding a test or example notebook for your work to `/tests` or `/examples`, then push your changes to your forked repository or branch and make a pull request to this repository. I'll review it and merge it or make changes/comments before merging.

While working on package development, you can install the package in "development mode" by running `python setup.py develop` (what I use) or `python -m pip install -e .` (what I think is now recommended) in the root directory of the repository ([more info here]([https://stackoverflow.com/questions/19048732/python-setup-py-develop-vs-install](https://packaging.python.org/en/latest/guides/distributing-packages-using-setuptools/#working-in-development-mode))). Again, I recommend doing all of this in a virtual environment.

Both [VS Code](https://code.visualstudio.com/docs/sourcecontrol/overview) and [JupyterLab](https://github.com/jupyterlab/jupyterlab-git) have git extensions that make this process easy.


