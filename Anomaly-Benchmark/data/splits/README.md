# Exact split indices

`split_indices.npz` contains five `int32` arrays:

- `train`: 0–31,999
- `buffer_1`: 32,000–36,095
- `calibration`: 36,096–44,191
- `buffer_2`: 44,192–48,287
- `test`: 48,288–59,999

The arrays are mutually disjoint, sorted, contiguous within each split, and together cover every sample index from 0 through 59,999 exactly once.
