# MLIP Profiling Results

Model: {'type': 'sevenn', 'name': '7net-0', 'path': None, 'cutoff': 5.0, 'backend': 'e3nn', 'num_elements': 89, 'channel': 128, 'lmax': 2, 'num_convolution_layer': 5, 'is_parity': False}

## Performance Summary

| System | Atoms | Time (ms) | QPS | ns/day |
|--------|-------|-----------|-----|--------|
| Cu_fcc_3x3x3_108atoms_108atoms | 108 | 53.86 ± 0.57 | 18.6 | 1.60 |
| Cu_fcc_5x5x5_500atoms_500atoms | 500 | 69.82 ± 1.35 | 14.3 | 1.24 |
| Cu_fcc_7x7x7_1372atoms_1372atoms | 1372 | 139.07 ± 0.08 | 7.2 | 0.62 |
| Cu_fcc_9x9x9_2916atoms_2916atoms | 2916 | 289.12 ± 0.17 | 3.5 | 0.30 |