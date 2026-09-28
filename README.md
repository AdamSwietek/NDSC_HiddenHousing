# NDSC Hidden Housing

Geospatial feature engineering for comparing Los Angeles County building footprints, parcel records, and assessor data across survey years.

## Project layout

- `src/`: data loading, geometry cleaning, parcel matching, snapshot metrics, change metrics, and pipeline code.
- `scripts/`: command-line pipeline entry points.
- `tests/`: unit and integration tests.
- `notebooks/methods/`: notebooks used to develop and evaluate identification methods.
- `data/`: local source and generated datasets. The entire directory is intentionally excluded from Git.

## Current method notebook

`notebooks/methods/01_floor_area_assessor_method.ipynb` develops Method 1 using agreement between 2017 LARIAC estimated floor area and assessor `SQFTmain` within use type.
