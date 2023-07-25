# Parsing Conventions and Capabilities

`magnetopy` can read in a variety of raw magnetic data and convert it to useful Python objects. The following sections describe what sort of file conventions are compatible with `magnetopy` and how the automatic parsing works.

## Sample Info and Moment Scaling

When creating a `Dataset` object, `magnetopy` will assume that all files within the given folder are from the same sample. Thus, the `Dataset.sample_info` attribute will be created by looking at the sample information in the header of the first file that it finds.

The sample information is also used to automatically scale the magnetic moment data in all dc measurements (i.e., all `MvsH`, `ZFC`, and `FC` objects within the dataset).

## .dat File Conventions

### Commented Files

Comments should be comma separated and can be in any order.

#### MvsH

- comments must include:
  - "MvsH" (case insensitive)
  - the nominal temperature written as "XX", "XX K", or "XX C" (values without a unit are assumed to be K)

#### ZFCFC, ZFC, FC

- comments must include:
  - "ZFC" or "FC" (case insensitive)
  - the nominal magnetic field strength written as "XX", "XX Oe", or "XX T" (values without a unit are assumed to be Oe)

### Uncommented Files

#### MvsH

#### ZFCFC, ZFC, FC
