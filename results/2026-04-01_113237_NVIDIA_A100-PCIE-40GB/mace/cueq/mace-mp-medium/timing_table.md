# MLIP Profiling Results

Model: {'type': 'mace', 'path': '/home/jeheon/Models/mace/mace-mp-0-medium.model', 'cutoff': 6.0, 'backend': 'cueq', 'heads': ['default'], 'num_elements': 89}

## Performance Summary

| System | Atoms | Time (ms) | QPS | ns/day |
|--------|-------|-----------|-----|--------|
| Cu_fcc_3x3x3_108atoms_108atoms | 108 | 45.42 ± 0.07 | 22.0 | 1.90 |
| Cu_fcc_5x5x5_500atoms_500atoms | 500 | 48.67 ± 0.06 | 20.5 | 1.78 |
| Cu_fcc_7x7x7_1372atoms_1372atoms | 1372 | 76.01 ± 0.30 | 13.2 | 1.14 |
| Cu_fcc_9x9x9_2916atoms_2916atoms | 2916 | 145.00 ± 0.31 | 6.9 | 0.60 |