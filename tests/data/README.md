## DC Measurements

### MvsH

| File         | Measurement | Nominal Fields (Oe)     | Nominal Temperatures (K) | Comments? | Description                                                                                      |
| ------------ | ----------- | ----------------------- | ------------------------ | --------- | ------------------------------------------------------------------------------------------------ |
| mvsh1.dat    | VSM         | -70000 - 70000 (scan)   | 2, 4, 6, 8, 10, 12, 300  | No        | all temperatures contain full loops (no virgin) except 300 K, which only contains a reverse scan |
| mvsh2.dat    | DC          | -70000 - 70000 (settle) | 5, 300                   | No        | all temperatures contain full loops (no virgin)                                                  |
| mvsh2a.dat   | DC          | -70000 - 70000 (settle) | 5                        | No        | 5 K data from mvsh2.dat                                                                          |
| mvsh2b.dat   | DC          | -70000 - 70000 (settle) | 300                      | No        | 300 K data from mvsh2.dat                                                                        |
| mvsh2c.dat   | DC          | -70000 - 70000 (settle) | 5, 300                   | Yes       | Same as mvsh2.dat but with comments in the data section                                          |
| mvsh3.dat    | DC          | -70000 - 70000 (settle) | 5                        | No        | full loop (no virgin); uncommon header length                                                    |
| mvsh4.dat    | DC          | -70000 - 70000 (settle) | 293                      | Yes       | full loop (no virgin)                                                                            |
| mvsh5.dat    | DC          | -70000 - 70000 (settle) | 293                      | Yes       | full loop (no virgin)                                                                            |
| mvsh5.rw.dat | n/a         | n/a                     | n/a                      | n/a       | Unprocessed data (voltage vs position) from mvsh5.dat                                            |
| mvsh6.dat    | DC          | -70000 - 70000 (settle) | 300                      | No        | full loop (no virgin), field correction with Pd_std1.dat                                         |
| mvsh7.dat    | DC          | -70000 - 70000 (settle) | 300                      | No        | full loop (no virgin)                                                                            |
| mvsh8.dat    | VSM         | -70000 - 70000 (scan)   | 2                        | No        | virgin, reverse, forward                                                                         |
| mvsh9.dat    | DC          | -70000 - 70000 (settle) | 2                        | No        | virgin, reverse, forward                                                                         |
| mvsh10.dat   | DC          | -70000 - 70000 (settle) | 5                        | No        | full loop (no virgin)                                                                            |
| mvsh11.dat   | VSM         | -70000 - 70000 (scan)   | 5                        | No        | virgin, reverse, forward                                                                         |
| Pd_std1.dat  | DC          | -70000 - 70000 (settle) | 300                      | No        | full loop (no virgin)                                                                            |

### ZFCFC

| File        | Measurement | Nominal Fields (Oe) | Nominal Temperatures (K) | Comments? | Description                                               |
| ----------- | ----------- | ------------------- | ------------------------ | --------- | --------------------------------------------------------- |
| zfcfc1.dat  | DC          | 100                 | 5 - 300 (scan)           | No        | ZFC 5 to 300 K, then temperature drop, then FC 5 to 300 K |
| zfcfc2.dat  | DC          | 100                 | 5 - 340 (scan)           | No        | ZFC 5 to 340 K, then temperature drop, then FC 5 to 340 K |
| zfcfc3.dat  | DC          | 100                 | 5 - 300 (scan)           | No        | ZFC 5 to 300 K, then FC 300 to 5 K                        |
| zfc4a.dat   | VSM         | 100                 | 5 - 310 (scan)           | Yes       | only ZFC at 100 Oe                                        |
| zfc4b.dat   | VSM         | 1000                | 5 - 310 (scan)           | Yes       | only ZFC at 1000 Oe                                       |
| fc4a.dat    | VSM         | 100                 | 310 - 5 (scan)           | Yes       | only FC at 100 Oe                                         |
| fc4b.dat    | VSM         | 1000                | 310 - 5 (scan)           | Yes       | only FC at 1000 Oe                                        |
| zfcfc4.dat  | VSM         | 100, 1000           | 5 - 310 (scan)           | Yes       | combines zfc4a, fc4a, zfc4b, fc4b                         |
| zfc5.dat    | DC          | 200                 | 2 - 300 (settle)         | No        | only ZFC at 200 Oe                                        |
| zfc5.rw.dat | n/a         | n/a                 | n/a                      | n/a       | Unprocessed data (voltage vs position) from zfc5.dat      |
| fc5.dat     | DC          | 200                 | 2 - 300 (settle)         | No        | only FC at 200 Oe                                         |
| fc5.rw.dat  | n/a         | n/a                 | n/a                      | n/a       | Unprocessed data (voltage vs position) from fc5.dat       |
| zfcfc6.dat  | DC          | 100                 | 5 - 300 (scan)           | No        | ZFC 5 to 300 K, then temperature drop, then FC 5 to 300 K |
| fc7.dat     | DC          | 100                 | 2 - 300 (scan)           | No        | only FC at 100 Oe                                         |
| zfc7.dat    | DC          | 100                 | 2 - 300 (scan)           | No        | only ZFC at 100 Oe                                        |
| fc8.dat     | DC          | 250                 | 2 - 300 (scan)           | No        | only FC at 250 Oe                                         |
| zfc8.dat    | DC          | 250                 | 2 - 300 (scan)           | No        | only ZFC at 250 Oe                                        |
| fc9.dat     | DC          | 500                 | 2 - 300 (scan)           | No        | only FC at 500 Oe                                         |
| zfc9.dat    | DC          | 500                 | 2 - 300 (scan)           | No        | only ZFC at 500 Oe                                        |
| fc10.dat    | DC          | 750                 | 2 - 300 (scan)           | No        | only FC at 750 Oe                                         |
| zfc10.dat   | DC          | 750                 | 2 - 300 (scan)           | No        | only ZFC at 750 Oe                                        |
| fc11.dat    | DC          | 1000                | 2 - 300 (scan)           | No        | only FC at 1000 Oe                                        |
| zfc11.dat   | DC          | 1000                | 2 - 300 (scan)           | No        | only ZFC at 1000 Oe                                       |
| fc12.dat    | DC          | 10000               | 2 - 300 (scan)           | No        | only FC at 10000 Oe                                       |
| zfc12.dat   | DC          | 10000               | 2 - 300 (scan)           | No        | only ZFC at 10000 Oe                                      |
| fc13.dat    | DC          | 40000               | 2 - 300 (scan)           | No        | only FC at 40000 Oe                                       |
| zfc13.dat   | DC          | 40000               | 2 - 300 (scan)           | No        | only ZFC at 40000 Oe                                      |

### Mixed Files

| File         | Measurement | Nominal Fields (Oe)   | Nominal Temperatures (K) | Comments? | Description                              |
| ------------ | ----------- | --------------------- | ------------------------ | --------- | ---------------------------------------- |
| dataset4.dat | VSM, DC     | 100, 1000, -70k - 70k | 5 - 310 (scan), 293      | Yes       | combines zfc4a, fc4a, zfc4b, fc4b, mvsh4 |

## Datasets

| Name     | Files                                                 | Notes                                                                                                         |
| -------- | ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| dataset1 | mvsh1.dat, zfcfc1.dat                                 |                                                                                                               |
| dataset2 | mvsh5.dat, mvsh5.rw.dat, zfcfc2.dat                   |                                                                                                               |
| dataset3 | mvsh6.dat, zfcfc4.dat                                 |                                                                                                               |
| dataset4 | mvsh9.dat, fc5.dat, fc5.rw.dat, zfc5.dat, zfc5.rw.dat |                                                                                                               |
| dataset5 | mvsh12.dat, zfcfc6.dat                                | From [Kirkpatrick, et al. Chem. Sci. 2023](https://pubs.rsc.org/en/content/articlelanding/2023/SC/D3SC02113K) |
| dataset6 | mvsh13.dat, zfc7 - zfc13, fc7 - fc13                  | From [Orlova, et al. JACS 2023](https://pubs.acs.org/doi/full/10.1021/jacs.3c08946#)                          |
