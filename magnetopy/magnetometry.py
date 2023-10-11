from __future__ import annotations
from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Any, Literal, Protocol

import matplotlib.pyplot as plt

from magnetopy.data_files import DatFile

from magnetopy.experiments.dc_experiment import DcExperiment
from magnetopy.experiments.mvsh import MvsH, plot_mvsh
from magnetopy.experiments.zfcfc import FC, ZFC, plot_zfcfc


@dataclass
class SampleInfo:
    """Information specific to the particular sample used for magnetic measurements.

    Attributes
    ----------
    material : str | None
        The material used for the sample. Possibly a chemical formula.
    comment : str | None
        Any comments about the sample.
    mass : float | None
        The mass of the sample in milligrams.
    volume : float | None
        The volume of the sample in milliliters or cubic centimeters.
    molecular_weight : float | None
        The molecular weight of the sample in grams per mole.
    size : float | None
        The size of the sample in millimeters.
    shape : str | None
        The shape of the sample.
    holder : str | None
        The type of sample holder used. Usually "quartz", "straw", or "brass".
    holder_detail : str | None
        Any additional details about the sample holder.
    offset : float | None
        The vertical offset of the sample holder in millimeters.
    eicosane_mass : float | None
        The mass of the eicosane in milligrams.
    diamagnetic_correction : float | None
        The diamagnetic correction in emu/mol.
    """

    material: str | None = None
    comment: str | None = None
    mass: float | None = None
    volume: float | None = None
    molecular_weight: float | None = None
    size: float | None = None
    shape: str | None = None
    holder: str | None = None
    holder_detail: str | None = None
    offset: float | None = None
    eicosane_mass: float | None = None
    diamagnetic_correction: float | None = None

    @classmethod
    def from_dat_file(
        cls, dat_file: str | Path | DatFile, eicosane_field_hack: bool = True
    ) -> SampleInfo:
        """
        Create a SampleInfo object from a .dat file (either a path to the file or a
        DatFile object). Sample information is extracted from the header of the .dat
        file.

        Parameters
        ----------
        dat_file : str | Path | DatFile
            The .dat file to read the sample information from.
        eicosane_field_hack : bool, optional
            For abuse of the Quantum Design sample info fields, where the "volume" and
            "size" fields are used to store the eicosane mass and diamagnetic
            correction, respectively. The default is True.

        Returns
        -------
        SampleInfo
        """
        if not isinstance(dat_file, DatFile):
            dat_file = DatFile(dat_file)
        sample = cls()
        for line in dat_file.header:
            category = line[0]
            if category != "INFO":
                continue
            if not line[1]:
                continue
            info = line[2]
            if info == "SAMPLE_MATERIAL":
                sample.material = line[1]
            elif info == "SAMPLE_COMMENT":
                sample.comment = line[1]
            elif info == "SAMPLE_MASS":
                sample.mass = float(line[1])
            elif info == "SAMPLE_VOLUME":
                sample.volume = float(line[1])
            elif info == "SAMPLE_MOLECULAR_WEIGHT":
                sample.molecular_weight = float(line[1])
            elif info == "SAMPLE_SIZE":
                sample.size = float(line[1])
            elif info == "SAMPLE_SHAPE":
                sample.shape = line[1]
            elif info == "SAMPLE_HOLDER":
                sample.holder = line[1]
            elif info == "SAMPLE_HOLDER_DETAIL":
                sample.holder_detail = line[1]
            elif info == "SAMPLE_OFFSET":
                sample.offset = float(line[1])

        if eicosane_field_hack:
            sample.eicosane_mass = sample.volume
            sample.diamagnetic_correction = sample.size
            sample.size, sample.volume = None, None
        return sample

    def as_dict(self) -> dict[str, float | str | None]:
        """Create a dictionary representation of the SampleInfo object.

        Returns
        -------
        dict[str, foat | str | None]
        """
        output = asdict(self)
        output["_class_"] = self.__class__.__name__
        return output


class Analysis(Protocol):
    """A protocol class representing an analysis of one or more magnetometry
    experiments.

    Attributes
    ----------
    results : Any
        The results of the analysis. For results more complicated than a simple value,
        it is recommended that a `dataclass` be used, and that the `dataclass` have an
        `as_dict` method.

    Notes
    -----
    While not yet enforced, it is strongly recommended that any class that implements
    this protocol have an `__init__` method which takes the following arguments:

    - `dataset`: a `Magnetometry` object
    - `parsing_args`: an object with attributes that specify how to parse the data;
    these arguments may be used as arguments to be passed to various `Magnetometry`
    methods. It is recommended that this object be a dataclass with an `as_dict`
    method.
    - `fitting_args`: an object with attributes that specify how to fit the data;
    these arguments may include, e.g., starting values, bounds, constraints, etc.
    It is recommended that this object be a dataclass with an `as_dict` method.

    The `__init__` method should perform the analysis and store the results in the
    `results` attribute.
    """

    def as_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the analysis. Should include the
        results of the analysis along with any parsing and fitting arguments used.

        The dictionary must include a field "_class_" with the name of the class of
        the analysis object (e.g. "MvsHAnalysis", "ZFCAnalysis", etc.).

        Returns
        -------
        dict[str, Any]
        """
        ...


class Magnetometry:
    """
    A class which contains magnetometry data for a single sample along with methods for
    parsing, processing, and analyzing the data. Note that "single sample" means that
    in situations in which a single _material_ is measured in multiple samples, each
    sample should be treated as a separate `Magnetometry` object.

    Parameters
    ----------
    path : str | Path
        The path to the directory containing the .dat files for the sample.
    sample_id : str, optional
        The sample ID. If not provided, the name of the directory containing the .dat
        files will be used.
    magnetic_data_scaling : str | list[str], optional
        Default `"auto"`. Instructions for scaling the magnetic moment. Options are
        "mass", "molar", "eicosane", "diamagnetic_correction", and "auto". If "auto" is
        specified, the scaling will be determined automatically based on the available
        sample information found in the first .dat file found in the directory.
    true_field_correction : str | Path, optional
        The path to a file containing the M vs. H data of a Pd standard sample and to
        be used for correcting the field of all `MvsH` objects. Note that this is a
        convenience method for situations in which all `MvsH` objects use the same
        sequence and can be corrected by the same file. Individual corrections can be
        performed by calling the `correct_field` method of the relevant `MvsH` object.
    parse_raw : bool, optional
        Default `False`. If `True`, any .rw.dat files in the directory will be parsed
        and the raw data will be stored in the `"raw_scan"` column of the corresponding
        `data` attribute of the relevant `DcExperiment` object.

    Attributes
    ----------
    sample_id : str
        The sample ID.
    files : list[DatFile]
        A list of the `DatFile` objects in the directory. Note that this includes both
        .dat and .rw.dat files.
    magnetic_data_scaling : list[str]
        A record of the scaling options used to scale the magnetic data. Options are
        "mass", "molar", "eicosane", and "diamagnetic_correction". Currently supported
        combinations are: `["mass"] | ["molar"] | ["molar", "eicosane"] | ["molar",
        "diamagnetic_correction"] | ["molar", "eicosane", "diamagnetic_correction"]`.
    sample_info : SampleInfo
        Information specific to the particular sample used for magnetic measurements.
        Note that this is determined by reading the first `DatFile` in `files`, and
        it is assumed that all `DatFile` objects in `files` are for the same
        sample.
    mvsh : list[MvsH]
        A list of the `MvsH` objects in the directory.
    zfc : list[ZFC]
        A list of the `ZFC` objects in the directory.
    fc : list[FC]
        A list of the `FC` objects in the directory.
    analyses : list[Analysis]
        A list of the analyses performed on the data.

    Notes
    -----
    Scaling of magnetic data occurs during object initialization, unless
    `magnetic_data_scaling` is set to `''`. If the user needs to overwrite the
    automatically scaled values (say, if the sample information from the .dat file
    needs to be altered), the user can do so by overwriting the relevant attributes of
    the `sample_info` attribute (e.g. `mass`, `molecular_weight`, etc.), then calling
    the `scale_dc_data` method.
    """

    class ExperimentNotFoundError(Exception):
        pass

    def __init__(
        self,
        path: str | Path,
        sample_id: str = "auto",
        magnetic_data_scaling: str | list[str] = "auto",
        true_field_correction: str | Path = "",
        parse_raw: bool = False,
    ) -> None:
        path = Path(path)
        self.sample_id = path.name if sample_id == "auto" else sample_id
        self.files = [
            DatFile(file, parse_raw) for file in path.rglob("*.dat") if file.is_file()
        ]
        self.magnetic_data_scaling = (
            [magnetic_data_scaling]
            if isinstance(magnetic_data_scaling, str)
            else magnetic_data_scaling
        )
        self.sample_info = SampleInfo.from_dat_file(self.files[0])
        self.mvsh = self.extract_mvsh()
        self.zfc = self.extract_zfc()
        self.fc = self.extract_fc()
        self.scale_dc_data()
        if true_field_correction:
            self.correct_field(true_field_correction)
        self.analyses: list[Analysis] = []

    def __str__(self) -> str:
        return f"Dataset({self.sample_id})"

    def __repr__(self) -> str:
        return f"Dataset({self.sample_id})"

    def extract_mvsh(
        self, eps: float = 0.001, min_samples: int = 10, ndigits: int = 0
    ) -> list[MvsH]:
        """Extracts all M vs. H experiments found within `files`. This is run
        during object initialization but is available to the user in the event that
        the default parsing parameters need to be changed for proper data extraction.

        See `magnetopy.experiments.mvsh.MvsH.get_all_in_file` for more information.

        Parameters
        ----------
        eps : float, optional
        min_samples : int, optional
        ndigits : int, optional

        Returns
        -------
        list[MvsH]
            The `MvsH` objects found in `files`.
        """
        mvsh_files = [
            dat_file
            for dat_file in self.files
            if "mvsh" in dat_file.experiments_in_file
        ]
        mvsh_objs: list[MvsH] = []
        for dat_file in mvsh_files:
            mvsh_objs.extend(MvsH.get_all_in_file(dat_file, eps, min_samples, ndigits))
        mvsh_objs.sort(key=lambda x: x.temperature)
        return mvsh_objs

    def extract_zfc(self, n_digits: int = 0) -> list[ZFC]:
        """Extracts all ZFC experiments found within `files`. This is run
        during object initialization but is available to the user in the event that
        the default parsing parameters need to be changed for proper data extraction.

        See `magnetopy.experiments.zfcfc.ZFC.get_all_in_file` for more information.

        Parameters
        ----------
        n_digits : int, optional

        Returns
        -------
        list[ZFC]
            The `ZFC` objects found in `files`.
        """
        zfc_files = [
            dat_file
            for dat_file in self.files
            if set(["zfc", "zfcfc"]).intersection(dat_file.experiments_in_file)
        ]
        zfc_objs: list[ZFC] = []
        for dat_file in zfc_files:
            zfc_objs.extend(ZFC.get_all_in_file(dat_file, n_digits))
        zfc_objs.sort(key=lambda x: x.field)
        return zfc_objs

    def extract_fc(self, n_digits: int = 0) -> list[FC]:
        """Extracts all FC experiments found within `files`. This is run
        during object initialization but is available to the user in the event that
        the default parsing parameters need to be changed for proper data extraction.

        See `magnetopy.experiments.zfcfc.FC.get_all_in_file` for more information.

        Parameters
        ----------
        n_digits : int, optional
            Default 0

        Returns
        -------
        list[FC]
            The `FC` objects found in `files`.
        """
        fc_files = [
            dat_file
            for dat_file in self.files
            if set(["fc", "zfcfc"]).intersection(dat_file.experiments_in_file)
        ]
        fc_objs: list[FC] = []
        for dat_file in fc_files:
            fc_objs.extend(FC.get_all_in_file(dat_file, n_digits))
        fc_objs.sort(key=lambda x: x.field)
        return fc_objs

    def scale_dc_data(self) -> None:
        """Scales the magnetic moment of all `DcExperiment` objects (i.e., `MvsH`,
        `ZFC`, and `FC` objects) in the `Magnetometry` object according to the scaling
        options specified in `magnetic_data_scaling` and the sample information in
        `sample_info`. This is run during object initialization but is available to
        the user in the event that either the defaul scaling options or the sample
        information need to be changed for proper data scaling.

        See `magnetopy.experiments.utils.scale_dc_data` for more information.
        """
        experiments: list[DcExperiment] = []
        experiments.extend(self.mvsh)
        experiments.extend(self.zfc)
        experiments.extend(self.fc)
        mass = (
            self.sample_info.mass
            if set(self.magnetic_data_scaling).intersection(["mass", "molar", "auto"])
            else 0
        )
        eicosane_mass = (
            self.sample_info.eicosane_mass
            if set(self.magnetic_data_scaling).intersection(["eicosane", "auto"])
            else 0
        )
        mol_weight = (
            self.sample_info.molecular_weight
            if set(self.magnetic_data_scaling).intersection(["molar", "auto"])
            else 0
        )
        diamagnetic_correction = (
            self.sample_info.diamagnetic_correction
            if set(self.magnetic_data_scaling).intersection(
                ["diamagnetic_correction", "auto"]
            )
            else 0
        )
        for experiment in experiments:
            experiment.scale_moment(
                mass, eicosane_mass, mol_weight, diamagnetic_correction
            )

    def correct_field(self, field_correction_file: str | Path) -> None:
        """A convenience method for correcting the field of all `MvsH` objects in the
        `Magnetometry` object using the same field correction file.

        See `magnetopy.experiments.mvsh.MvsH.correct_field` for more information.

        Parameters
        ----------
        field_correction_file : str | Path
            The path to the field correction file.
        """
        for experiment in self.mvsh:
            experiment.correct_field(field_correction_file)

    def get_mvsh(self, temperature: float) -> MvsH:
        """Get the `MvsH` object at the specified temperature.

        Parameters
        ----------
        temperature : float
            Temperature in Kelvin.

        Returns
        -------
        MvsH
            The `MvsH` object at the specified temperature.

        Raises
        ------
        self.ExperimentNotFoundError
            If no `MvsH` object is found at the specified temperature.
        """
        for mvsh in self.mvsh:
            if mvsh.temperature == temperature:
                return mvsh
        raise self.ExperimentNotFoundError(
            f"No MvsH experiment found at temperature {temperature} K"
        )

    def get_zfc(self, field: float) -> ZFC:
        """Get the `ZFC` object at the specified field.

        Parameters
        ----------
        field : float
            Field in Oe.

        Returns
        -------
        ZFC
            The `ZFC` object at the specified field.

        Raises
        ------
        self.ExperimentNotFoundError
            If no `ZFC` object is found at the specified field.
        """
        for zfc in self.zfc:
            if zfc.field == field:
                return zfc
        raise self.ExperimentNotFoundError(
            f"No ZFC experiment found at field {field} Oe"
        )

    def get_fc(self, field: float) -> FC:
        """Get the `FC` object at the specified field.

        Parameters
        ----------
        field : float
            Field in Oe.

        Returns
        -------
        FC
            The `FC` object at the specified field.

        Raises
        ------
        self.ExperimentNotFoundError
            If no `FC` object is found at the specified field.
        """
        for fc in self.fc:
            if fc.field == field:
                return fc
        raise self.ExperimentNotFoundError(
            f"No FC experiment found at field {field} Oe"
        )

    def add_analysis(self, analysis: Analysis | list[Analysis]) -> None:
        """Add an analysis to the `Magnetometry` object.

        See `magnetopy.magnetometry.Analysis` for more information.

        Parameters
        ----------
        analysis : Analysis | list[Analysis]
            An instance of a class (or list of such instances) that implements the
            `Analysis` protocol.

        """
        if isinstance(analysis, list):
            self.analyses.extend(analysis)
        else:
            self.analyses.append(analysis)

    def as_dict(self) -> dict[str, Any]:
        """Create a dictionary representation of the `Magnetometry` object.

        Returns
        -------
        dict[str, Any]
        """
        return {
            "_class_": self.__class__.__name__,
            "sample_id": self.sample_id,
            "files": self.files,
            "sample_info": self.sample_info,
            "mvsh": self.mvsh,
            "zfc": self.zfc,
            "fc": self.fc,
            "analyses": self.analyses,
        }

    def plot_mvsh(
        self,
        temperatures: float | list[float] | None = None,
        normalized: bool = False,
        segment: Literal["", "virgin", "forward", "reverse", "loop"] = "",
        colors: str | list[str] = "auto",
        labels: str | list[str] | None = "auto",
        title: str = "",
        **kwargs,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot the M vs. H data in the `Magnetometry` object. If `temperatures` is
        `None`, all `MvsH` objects will be plotted. Otherwise, only the `MvsH` objects
        at the specified temperatures will be plotted.


        Parameters
        ----------
        temperatures : float | list[float] | None, optional
            Default `None`. The temperatures at which to plot the M vs. H data. If
            `None`, all `MvsH` objects will be plotted. Otherwise, only the `MvsH`
            objects at the specified temperatures will be plotted.
        normalized : bool, optional
            If `True`, the magnetization will be normalized to the maximum value, by
            default False.
        segment : {"", "virgin", "forward", "reverse", "loop"}, optional
            If a segment is given, only that segment will be plotted, by default "".
        colors : str | list[str], optional
            A list of colors corresponding to the `MvsH` objects in `mvsh`, by default
            "auto". If "auto" and `mvsh` is a single `MvsH` object, the color will be
            black. If "auto" and `mvsh` is a list of `MvsH` objects with different
            temperatures, the colors will be a linear gradient from blue to red. If
            "auto" and `mvsh` is a list of `MvsH` objects with the same temperature, the
            colors will be the default `matplotlib` colors.
        labels : str | list[str] | None, optional
            The labels to assign the `MvsH` objects in the axes legend, by default "auto".
            If "auto", the labels will be the `temperature` of the `MvsH` objects.
        title : str, optional
            The title of the plot, by default "".
        **kwargs
            Keyword arguments mostly meant to affect the plot style. See
            `magnetopy.experiments.plot_utils.handle_options` for details.

        Returns
        -------
        tuple[plt.Figure, plt.Axes]
            The figure and axes objects of the plot.
        """
        if temperatures is None:
            return plot_mvsh(
                self.mvsh, normalized, segment, colors, labels, title, **kwargs
            )
        temperatures = (
            [temperatures] if isinstance(temperatures, float) else temperatures
        )
        mvsh = [self.get_mvsh(temperature) for temperature in temperatures]
        return plot_mvsh(mvsh, normalized, segment, colors, labels, title, **kwargs)

    def plot_zfcfc(
        self,
        fields: float | list[float] | None = None,
        y_val: Literal["moment", "chi", "chi_t"] = "moment",
        normalized: bool = False,
        colors: str | list[str] = "auto",
        labels: str | list[str] | None = "auto",
        title: str = "",
        **kwargs,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot the ZFC/FC data in the `Magnetometry` object. If `fields` is `None`,
        all `ZFC` and `FC` objects will be plotted. Otherwise, only the `ZFC` and `FC`
        objects at the specified fields will be plotted.

        Parameters
        ----------
        fields : float | list[float] | None, optional
            Default `None`. The fields at which to plot the ZFC/FC data. If `None`, all
            `ZFC` and `FC` objects will be plotted. Otherwise, only the `ZFC` and `FC`
            objects at the specified fields will be plotted.
        normalized : bool, optional
            If `True`, the magnetization will be normalized to the maximum value, by
            default False.
        colors : str | list[str], optional
            A list of colors corresponding to the `ZFC`/`FC` pairs, by default "auto". If
            "auto" and there is a single pair, the color will be black. If "auto" and
            there is a list of pairs with different fields, the colors will be a linear
            gradient from purple to green. If "auto" and the list of pairs is at the same
            field, the colors will be the default `matplotlib` colors.
        labels : str | list[str] | None, optional
            The labels to assign the `ZFC`/`FC` pair in the axes legend, by default "auto".
            If "auto", the label will be the `field` of the `MvsH` object.
        title : str, optional
            The title of the plot, by default "".
        **kwargs
            Keyword arguments mostly meant to affect the plot style. See `handle_options`
            for details.

        Returns
        -------
        tuple[plt.Figure, plt.Axes]
            The figure and axes objects of the plot.
        """
        if fields is None:
            return plot_zfcfc(
                self.zfc, self.fc, y_val, normalized, colors, labels, title, **kwargs
            )
        zfc = [self.get_zfc(field) for field in fields]
        fc = [self.get_fc(field) for field in fields]
        return plot_zfcfc(zfc, fc, y_val, normalized, colors, labels, title, **kwargs)

    def as_json(self, indent: int = 0) -> str:
        """Create a JSON representation of the `Magnetometry` object.

        Parameters
        ----------
        indent : int, optional
            Default 0. The number of spaces to indent the JSON.

        Returns
        -------
        str
        """

        return json.dumps(self, default=lambda x: x.as_dict(), indent=indent)

    def create_report(
        self, directory: str | Path | None = None, overwrite: bool = False
    ) -> None:
        """Create a JSON report of the `Magnetometry` object.

        Parameters
        ----------
        directory : str | Path | None, optional
            Default `None`. The directory to write the report to. If `None`, the report
            will be written to the directory containing the .dat files.
        overwrite : bool, optional
            Default `False`. Whether to overwrite the report if it already exists. If
            `False` and the report already exists, the user will be prompted to
            overwrite the report.
        """
        if directory:
            directory = Path(directory)
        else:
            directory = self.files[0].local_path.parent
        path = directory / f"{self.sample_id}.json"
        if path.exists() and overwrite is False:
            reponse = input(f"File {path} already exists. Overwrite? [y/N] ")
            if reponse.lower() != "y":
                print("Aborting report creation.")
                return
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.as_json(indent=4))
        print(f"Report written to {path}")
