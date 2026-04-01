# MLIP Profiling Results

Model: {'type': 'sevenn', 'name': '7net-0', 'path': None, 'cutoff': 5.0, 'backend': 'cueq', 'num_elements': 89, 'channel': 128, 'lmax': 2, 'num_convolution_layer': 5, 'is_parity': False}

## Performance Summary

| System | Atoms | Time (ms) | QPS | ns/day |
|--------|-------|-----------|-----|--------|
| Cu_fcc_3x3x3_108atoms_108atoms | 108 | 38.30 ± 0.03 | 26.1 | 2.26 |
| Cu_fcc_5x5x5_500atoms_500atoms | 500 | 46.33 ± 0.04 | 21.6 | 1.86 |
| Cu_fcc_7x7x7_1372atoms_1372atoms | 1372 | 51.98 ± 0.11 | 19.2 | 1.66 |
| Cu_fcc_9x9x9_2916atoms_2916atoms | 2916 | 72.14 ± 0.04 | 13.9 | 1.20 |