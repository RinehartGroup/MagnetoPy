## DC Measurements

### MvsH

| File         | Measurement | Nominal Fields (Oe)     | Nominal Temperatures (K) | Comments? | Description                                                                                      |
| ------------ | ----------- | ----------------------- | ------------------------ | --------- | ------------------------------------------------------------------------------------------------ |
| mvsh1.dat    | VSM         | -70000 - 70000 (scan)   | 2, 4, 6, 8, 10, 12, 300  | No        | all temperatures contain full loops (no virgin) except 300 K, which only contains a reverse scan |
| mvsh2.dat    | DC          | -70000 - 70000 (settle) | 5, 300                   | No        | all temperatures contain full loops (no virgin)                                                  |
| mvsh2a.dat   | DC          | -70000 - 70000 (settle) | 5                        | No        | 5 K data from mvsh2.dat                                                                          |
| mvsh2b.dat   | DC          | -70000 - 70000 (settle) | 300                      | No        | 300 K data from mvsh2.dat                                                                        |
| mvsh3.dat    | DC          | -70000 - 70000 (settle) | 5                        | No        | full loop (no virgin); uncommon header length                                                    |
| mvsh4.dat    | DC          | -70000 - 70000 (settle) | 293                      | Yes       | full loop (no virgin)                                                                            |
| mvsh5.dat    | DC          | -70000 - 70000 (settle) | 293                      | Yes       | full loop (no virgin)                                                                            |
| mvsh5.rw.dat | n/a         | n/a                     | n/a                      | n/a       | Unprocessed data (voltage vs position) from mvsh5.dat                                            |
| mvsh6.dat    | DC          | -70000 - 70000 (settle) | 300                      | No        | full loop (no virgin), field correction with Pd_std1 with no interpolation                       |
| mvsh7.dat    | DC          | -70000 - 70000 (settle) | 300                      | No        | full loop (no virgin), field correction with Pd_std1 with interpolation                          |
| mvsh8.dat    | VSM         | -70000 - 70000 (scan)   | 2                        | No        | virgin, reverse, forward                                                                         |
| mvsh9.dat    | DC          | -70000 - 70000 (settle) | 2                        | No        | virgin, reverse, forward                                                                         |
| mvsh10.dat   | DC          | -70000 - 70000 (settle) | 5                        | No        | full loop (no virgin)                                                                            |
| mvsh11.dat   | VSM         | -70000 - 70000 (scan)   | 5                        | No        | virgin, reverse, forward                                                                         |
| Pd_std1.dat  | DC          | -70000 - 70000 (settle) | 300                      | No        | full loop (no virgin)                                                                            |

### ZFCFC

| File       | Measurement | Nominal Fields (Oe) | Nominal Temperatures (K) | Comments? | Description                                               |
| ---------- | ----------- | ------------------- | ------------------------ | --------- | --------------------------------------------------------- |
| zfcfc1.dat | DC          | 100                 | 5 - 300 (scan)           | No        | ZFC 5 to 300 K, then temperature drop, then FC 5 to 300 K |
| zfcfc2.dat | DC          | 100                 | 5 - 340 (scan)           | No        | ZFC 5 to 340 K, then temperature drop, then FC 5 to 340 K |
| zfcfc3.dat | DC          | 100                 | 5 - 300 (scan)           | No        | ZFC 5 to 300 K, then FC 300 to 5 K                        |
| zfc4a.dat  | VSM         | 100                 | 5 - 310 (scan)           | Yes       | only ZFC at 100 Oe                                        |
| zfc4b.dat  | VSM         | 1000                | 5 - 310 (scan)           | Yes       | only ZFC at 1000 Oe                                       |
| fc4a.dat   | VSM         | 100                 | 310 - 5 (scan)           | Yes       | only FC at 100 Oe                                         |
| fc4b.dat   | VSM         | 1000                | 310 - 5 (scan)           | Yes       | only FC at 1000 Oe                                        |
| zfcfc4.dat | VSM         | 100, 1000           | 5 - 310 (scan)           | Yes       | combines zfc4a, fc4a, zfc4b, fc4b                         |

### Mixed Files

| File         | Measurement | Nominal Fields (Oe)   | Nominal Temperatures (K) | Comments? | Description                              |
| ------------ | ----------- | --------------------- | ------------------------ | --------- | ---------------------------------------- |
| dataset4.dat | VSM, DC     | 100, 1000, -70k - 70k | 5 - 310 (scan), 293      | Yes       | combines zfc4a, fc4a, zfc4b, fc4b, mvsh4 |
