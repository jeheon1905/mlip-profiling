# MLIP Profiling Results

Model: {'type': 'esen', 'name': 'esen-sm-conserving-all-omol', 'cutoff': 6, 'max_neighbors': 300, 'external_graph_gen': False, 'tf32': False, 'compile': False}

## Performance Summary

| System | Atoms | Time (ms) | QPS | ns/day |
|--------|-------|-----------|-----|--------|
| Cu_fcc_3x3x3_108atoms_108atoms | 108 | 58.26 ± 0.07 | 17.2 | 1.48 |
| Cu_fcc_5x5x5_500atoms_500atoms | 500 | 184.79 ± 0.52 | 5.4 | 0.47 |
| Cu_fcc_7x7x7_1372atoms_1372atoms | 1372 | 476.50 ± 1.40 | 2.1 | 0.18 |