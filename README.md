# NDSC Hidden Housing

Geospatial feature engineering for comparing Los Angeles County building footprints, parcel records, and assessor data across survey years.

## Project layout

- `src/`: data loading, geometry cleaning, parcel matching, snapshot metrics, change metrics, and pipeline code.
- `scripts/`: command-line pipeline entry points.
- `tests/`: unit and integration tests.
- `notebooks/methods/`: notebooks used to develop and evaluate identification methods.
- `data/`: local source and generated datasets. The entire directory is intentionally excluded from Git.

## Method notebooks

- `notebooks/methods/01_floor_area_assessor_method.ipynb`: Method 1, agreement between 2017 LARIAC estimated floor area and assessor `SQFTmain` within use type.
- `notebooks/methods/02_structural_survival_method.ipynb`: Method 2, structural survival 2014–2017 and distance from known UDUs.
- `notebooks/methods/03_udu_propensity_method.ipynb`: Method 3, cross-fitted XGBoost UDU propensity.
- `notebooks/methods/04_method2_vs_method3.ipynb`: compares the methods and writes the combined AIN-level table.

[`notebooks/methods/DATA_DICTIONARY.md`](notebooks/methods/DATA_DICTIONARY.md) describes every column of the combined table and how each method works.
