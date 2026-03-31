# MLIP Profiling Results

Model: {'type': 'mace', 'path': '/home/jeheon/Models/mace/mace-mp-0-small.model', 'cutoff': 6.0, 'backend': 'cueq', 'heads': ['default'], 'num_elements': 89}

## Performance Summary

| System | Atoms | Time (ms) | QPS | ns/day |
|--------|-------|-----------|-----|--------|
| Cu_fcc_3x3x3_108atoms_108atoms | 108 | 36.54 ± 0.04 | 27.4 | 2.36 |
| Cu_fcc_5x5x5_500atoms_500atoms | 500 | 39.46 ± 0.06 | 25.3 | 2.19 |
| Cu_fcc_7x7x7_1372atoms_1372atoms | 1372 | 71.92 ± 0.26 | 13.9 | 1.20 |
| Cu_fcc_9x9x9_2916atoms_2916atoms | 2916 | 134.33 ± 0.83 | 7.4 | 0.64 |