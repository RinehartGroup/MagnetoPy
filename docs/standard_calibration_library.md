# Standard Calibration Repository

MagnetoPy can be used to perform field corrections on M vs. H experiments and, in the future, may also implement manual background subtraction methods for dc measurements. Each magnetometer will have its own calibration files. Research groups may choose to maintain a repository of these files to serve as a single source of truth for their instrument and so that outside groups and reviewers have the necessary access to reproduce the results.

## Contents

The Rinehart group's repository is located in [MagnetoPyCalibration](https://github.com/RinehartGroup/MagnetoPyCalibration).

In general, a calibration library must include a the files containing calibraiton data and a json file describing the files.

"calibration.json" has the following structure:

```json
{
  "moment": "moment.dat",
  "mvsh": {
    "sequence_1": "mvsh_seq1.dat",
    "sequence_2": "mvsh_seq2.dat"
  }
}
```

The actual files are stored in the directory "calibration_files". It's assumed that, when needed, a .rw.dat file containing the unprocessed data is also stored in this folder (with the default convention of having the same name as the .dat file, but with the .rw.dat extension).

Only two categories of calibration files are currently supported.

1.  "moment"

    Used for the determination of the [system-specific calibration factor](https://qdusa.com/siteDocs/appNotes/1500-023.pdf). There should only ever need to be one of these files.

2.  "mvsh"

    Used for [determining the "true field" during an M vs H sequence](https://qdusa.com/siteDocs/appNotes/1500-021.pdf). Per the application note, all measurements should be collected at 298 K. The temperature of the calibration data does not need to match the temperature of the target data, but the sequence of fields should be the exact same.

    The "mvsh" keys are the names of the sequences. These names are referred to by `magnetopy` for easy access to the correct calibration data. See the [field correction example](../../examples/mvsh/#correcting-the-field-for-flux-trapping) for more details.
