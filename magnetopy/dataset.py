from __future__ import annotations
from dataclasses import dataclass

from magnetopy.data_files import DatFile


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
