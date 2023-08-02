from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


from magnetopy.data_files import DatFile
from magnetopy.experiments import FC, ZFC, MvsH, DcExperiment


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

    Methods
    -------
    from_dat_file(dat_file: str | DatFile, eicosane_field_hack: bool = True) -> SampleInfo
        Create a SampleInfo object from a .dat file (either a path to the file or a
        DatFile object).
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
        cls, dat_file: str | DatFile, eicosane_field_hack: bool = True
    ) -> SampleInfo:
        """
        Create a SampleInfo object from a .dat file (either a path to the file or a
        DatFile object).

        Parameters
        ----------
        dat_file : str | DatFile
            The .dat file to read the sample information from.
        eicosane_field_hack : bool, optional
            For abuse of the Quantum Design sample info fields, where the "volume" and
            "size" fields are used to store the eicosane mass and diamagnetic
            correction, respectively. The default is True.
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


class Dataset:
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
        self.dat_files = [
            DatFile(file, parse_raw) for file in path.rglob("*.dat") if file.is_file()
        ]
        self.magnetic_data_scaling = (
            [magnetic_data_scaling]
            if isinstance(magnetic_data_scaling, str)
            else magnetic_data_scaling
        )
        self.sample_info = SampleInfo.from_dat_file(self.dat_files[0])
        self.mvsh = self.extract_mvsh()
        self.zfc = self.extract_zfc()
        self.fc = self.extract_fc()
        self.scale_dc_data()
        if true_field_correction:
            self.correct_field(true_field_correction)

    def __str__(self) -> str:
        return f"Dataset({self.sample_id})"

    def __repr__(self) -> str:
        return f"Dataset({self.sample_id})"

    def extract_mvsh(
        self, eps: float = 0.001, min_samples: int = 10, ndigits: int = 0
    ) -> list[MvsH]:
        mvsh_files = [
            dat_file
            for dat_file in self.dat_files
            if "mvsh" in dat_file.experiments_in_file
        ]
        mvsh_objs: list[MvsH] = []
        for dat_file in mvsh_files:
            mvsh_objs.extend(MvsH.get_all_in_file(dat_file, eps, min_samples, ndigits))
        mvsh_objs.sort(key=lambda x: x.temperature)
        return mvsh_objs

    def extract_zfc(self, n_digits: int = 0) -> list[ZFC]:
        zfc_files = [
            dat_file
            for dat_file in self.dat_files
            if set(["zfc", "zfcfc"]).intersection(dat_file.experiments_in_file)
        ]
        zfc_objs: list[ZFC] = []
        for dat_file in zfc_files:
            zfc_objs.extend(ZFC.get_all_in_file(dat_file, n_digits))
        zfc_objs.sort(key=lambda x: x.field)
        return zfc_objs

    def extract_fc(self, n_digits: int = 0) -> list[FC]:
        fc_files = [
            dat_file
            for dat_file in self.dat_files
            if set(["fc", "zfcfc"]).intersection(dat_file.experiments_in_file)
        ]
        fc_objs: list[FC] = []
        for dat_file in fc_files:
            fc_objs.extend(FC.get_all_in_file(dat_file, n_digits))
        fc_objs.sort(key=lambda x: x.field)
        return fc_objs

    def scale_dc_data(self) -> None:
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
        for experiment in self.mvsh:
            experiment.correct_field(field_correction_file)

    def get_mvsh(self, temperature: float) -> MvsH:
        for mvsh in self.mvsh:
            if mvsh.temperature == temperature:
                return mvsh
        raise self.ExperimentNotFoundError(
            f"No MvsH experiment found at temperature {temperature} K"
        )

    def get_zfc(self, field: float) -> ZFC:
        for zfc in self.zfc:
            if zfc.field == field:
                return zfc
        raise self.ExperimentNotFoundError(
            f"No ZFC experiment found at field {field} Oe"
        )

    def get_fc(self, field: float) -> FC:
        for fc in self.fc:
            if fc.field == field:
                return fc
        raise self.ExperimentNotFoundError(
            f"No FC experiment found at field {field} Oe"
        )
