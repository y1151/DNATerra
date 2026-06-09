# DNATerra Input Data

This directory contains the built-in simulator noise library. The library includes 17 samples, with noise profiles characterized from public sequencing datasets.

These profiles are empirical dataset-level profiles, not mechanistic models of synthesis, PCR or sequencing processes. Sample names that contain PCR conditions identify the measured dataset only. They cannot be used to derive unmeasured PCR cycles or to automatically predict another PCR condition. Use each profile as an independent `--method`, or create a new profile from your own data with the self-update workflow.

Each built-in sample is represented by four `.npz` files:

- `read_coverage_depth_<SampleID>.npz`
- `per_position_error_rates_<SampleID>.npz`
- `error_bias_<SampleID>.npz`
- `error_bias_kmer_<SampleID>.npz`

Use the `Sample ID` value as the simulator `--method`.

| Sample ID | Synthesis Method | Sequencing method | Number of Reference Sequences | Number of Sequencing Reads | Reference Sequence Length (bp) | Loss Rate | Average Read Depth | Coefficient of Variation | Read Counts Range | Recommended Distribution | Substitution Error Rate (10^-3 nt^-1) | Insertion Error Rate (10^-3 nt^-1) | Deletion Error Rate (10^-3 nt^-1) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |
| PCR_15c_Genscript_GCall | Electrochemical | Illumina | 12472 | 3432862 | 102 | 0.0024 | 133.724 | 1 | [1, 825] | Exponential | 5.770 | 0.395 | 10.400 |
| PCR_30c_Genscript_GCall | Electrochemical | Illumina | 12472 | 2591899 | 102 | 0.5361 | 215.989 | 1.4034 | [1, 1888] | Gamma | 8.510 | 0.375 | 10.100 |
| PE-YAB | Inkjet | Illumina | 153335 | 133821570 | 183 | 0 | 423.564 | 0.251 | [7, 2541] | Lognormal | 16.900 | 0.114 | 4.350 |
| id20 | Inkjet | Illumina | 220948 | 25651716 | 177 | 0.000009 | 115.904 | 0.5433 | [1, 1161] | Beta | 4.610 | 2.960 | 3.300 |
| P30 | Inkjet | Illumina | 11520 | 22661497 | 180 | 0 | 947.199 | 1.3407 | [10, 41648] | Lognormal | 0.888 | 0.103 | 2.690 |
| Aging_0a_Twist_GCall | Inkjet | Illumina | 12000 | 2334581 | 108 | 0.0007 | 95.726 | 0.2989 | [1, 235] | Normal | 5.730 | 0.040 | 0.421 |
| Aging_7d_Twist_GCall | Inkjet | Illumina | 12000 | 2374514 | 108 | 0.0009 | 97.123 | 0.3188 | [1, 251] | Normal | 7.370 | 0.042 | 0.506 |
| phix_Twist_GCall_aging | Inkjet | Illumina | 12000 | 539879 | 108 | 0.0061 | 11.273 | 0.4360 | [1, 39] | Nbinom | 7.100 | 0.039 | 0.464 |
| PCR_15c_Twist_GCall | Inkjet | Illumina | 12000 | 1973913 | 108 | 0.0001 | 81.069 | 0.2855 | [1, 188] | Nbinom | 3.260 | 0.038 | 0.370 |
| PCR_30c_Twist_GCall | Inkjet | Illumina | 12000 | 1492448 | 108 | 0.0003 | 61.181 | 0.3099 | [1, 156] | Normal | 5.340 | 0.040 | 0.460 |
| PCR_45c_Twist_GCall | Inkjet | Illumina | 12000 | 1379700 | 108 | 0.0019 | 56.787 | 0.3245 | [1, 158] | Normal | 6.630 | 0.042 | 0.479 |
| PCR_60c_Twist_GCall | Inkjet | Illumina | 12000 | 2266172 | 108 | 0.0023 | 91.119 | 0.3243 | [1, 270] | Normal | 8.670 | 0.045 | 0.568 |
| PCR_75c_Twist_GCall | Inkjet | Illumina | 12000 | 1714804 | 108 | 0.0028 | 70.416 | 0.3517 | [1, 193] | Normal | 10.300 | 0.048 | 0.635 |
| PCR_90c_Twist_GCall | Inkjet | Illumina | 12000 | 1487430 | 108 | 0.0041 | 61.354 | 0.3735 | [1, 229] | Normal | 11.500 | 0.049 | 0.670 |
| phix_Twist_GCall_PCR | Inkjet | Illumina | 12000 | 931393 | 109 | 0.0007 | 21.233 | 0.3845 | [1, 67] | Nbinom | 7.860 | 0.041 | 0.516 |
| PCR_30c_Twist_GCfix | Inkjet | Illumina | 12000 | 1501272 | 110 | 0.0008 | 61.654 | 0.2752 | [1, 253] | Normal | 5.200 | 0.049 | 0.422 |
| I18-S3-R1-001 | Photochemical | Illumina | 16383 | 29303904 | 60 | 0 | 258.026 | 0.5503 | [2, 993] | Weibull | 11.200 | 8.280 | 17.100 |

## Test Sequences

The `seq_n*_l150.fasta` files are randomly generated DNA sequences for simulator testing. The `n*` value gives the number of 150 bp reference sequences in the file.

## Self-update Test Data

The following files are included to test the self-update workflow:

- `test.fasta`: reference sequences from a biochemical experiment
- `test_1.fq`: the first 2,500 sequencing reads from the same experiment
- `test_2.fq`: the first 2,500 paired-end sequencing reads from the same experiment
