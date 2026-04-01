# MLIP Profiling Results

Model: {'type': 'mace', 'path': '/home/jeheon/Models/mace/mace-mp-0-medium.model', 'cutoff': 6.0, 'backend': 'e3nn', 'heads': ['Default'], 'num_elements': 89}

## Performance Summary

| System | Atoms | Time (ms) | QPS | ns/day |
|--------|-------|-----------|-----|--------|
| Cu_fcc_3x3x3_108atoms_108atoms | 108 | 40.18 ± 0.02 | 24.9 | 2.15 |
| Cu_fcc_5x5x5_500atoms_500atoms | 500 | 106.47 ± 0.05 | 9.4 | 0.81 |
| Cu_fcc_7x7x7_1372atoms_1372atoms | 1372 | 286.57 ± 0.16 | 3.5 | 0.30 |