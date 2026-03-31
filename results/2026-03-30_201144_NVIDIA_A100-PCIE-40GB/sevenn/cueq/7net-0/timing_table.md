# MLIP Profiling Results

Model: {'type': 'sevenn', 'name': '7net-0', 'path': None, 'cutoff': 5.0, 'backend': 'cueq', 'num_elements': 89, 'channel': 128, 'lmax': 2, 'num_convolution_layer': 5, 'is_parity': False}

## Performance Summary

| System | Atoms | Time (ms) | QPS | ns/day |
|--------|-------|-----------|-----|--------|
| Cu_fcc_3x3x3_108atoms_108atoms | 108 | 38.81 ± 0.05 | 25.8 | 2.23 |
| Cu_fcc_5x5x5_500atoms_500atoms | 500 | 46.80 ± 0.05 | 21.4 | 1.85 |
| Cu_fcc_7x7x7_1372atoms_1372atoms | 1372 | 52.24 ± 0.02 | 19.1 | 1.65 |
| Cu_fcc_9x9x9_2916atoms_2916atoms | 2916 | 72.69 ± 0.06 | 13.8 | 1.19 |