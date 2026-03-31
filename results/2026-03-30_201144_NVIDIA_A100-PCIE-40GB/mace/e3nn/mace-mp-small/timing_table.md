# MLIP Profiling Results

Model: {'type': 'mace', 'path': '/home/jeheon/Models/mace/mace-mp-0-small.model', 'cutoff': 6.0, 'backend': 'e3nn', 'heads': ['Default'], 'num_elements': 89}

## Performance Summary

| System | Atoms | Time (ms) | QPS | ns/day |
|--------|-------|-----------|-----|--------|
| Cu_fcc_3x3x3_108atoms_108atoms | 108 | 32.64 ± 0.02 | 30.6 | 2.65 |
| Cu_fcc_5x5x5_500atoms_500atoms | 500 | 47.45 ± 0.01 | 21.1 | 1.82 |
| Cu_fcc_7x7x7_1372atoms_1372atoms | 1372 | 123.33 ± 0.04 | 8.1 | 0.70 |
| Cu_fcc_9x9x9_2916atoms_2916atoms | 2916 | 259.93 ± 0.34 | 3.8 | 0.33 |