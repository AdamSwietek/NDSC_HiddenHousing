# Potential non-UDU methods: data dictionary

`all_methods_potential_nonudu_2017.csv` combines three methods for identifying **potential non-UDUs**. It has one row per assessor parcel (AIN), with 2017 as the reference year. A potential non-UDU is a parcel whose building and assessor records look least like a reported unpermitted dwelling unit (UDU). The flag is a screening signal, not a verified finding.

- **File:** `data/03_processed/method_comparison/all_methods_potential_nonudu_2017.csv`. It is stored locally only, because `data/` is excluded from Git. It has about 553,000 rows and 38 columns (150 MB).
- **Produced by:** [04_method2_vs_method3.ipynb](04_method2_vs_method3.ipynb). This notebook joins the AIN-level score files that notebooks 01–03 save in `data/02_intermediate/method_scores/`.
- **To regenerate:** run [01](01_floor_area_assessor_method.ipynb), [02](02_structural_survival_method.ipynb) and [03](03_udu_propensity_method.ipynb), then [04](04_method2_vs_method3.ipynb).
- **Current file:** built on 29 September 2026 from 36 of 37 city tiles, while the city build was still running. Regenerate it once the build completes.

## Conventions

- **AIN:** read `AIN` as text. It is a 10-digit Assessor Identification Number.
- **Flags:** columns ending in `potential_nonudu`, and `KnownUDU`, `method2_structural_screen`, `method2_check_*` and `method2_udu_like`, are `True` or `False`. A blank means the method did not score that AIN, which is different from `False`.
- **Score direction:** the scores do not all point the same way; each column below states its direction. Ranks are percentile ranks from 0 to 1.
- **Known UDUs:** AINs listed in the city UDU report of 10 February 2026 (`data/01_source/UDU Report 02_10_2026.csv`). Methods 2 and 3 never flag them. Method 1 does not exclude them: 5 are flagged by its residual screen and 24 by its ratio screen, so filter on `KnownUDU`.
- **Eligible parcels:** single-AIN parcel sites with a positive assessor `SQFTmain` and a positive LARIAC estimated floor area. They must also be in one of six use types that have enough known UDUs: Single Family Residence; Double, Duplex, or Two Units; Three Units; Four Units; Five or More Units or Apartments; and Store Combination. 63 AINs are scored by Method 1 only, and their identity columns are blank.
- **LARIAC estimated floor area:** the sum over a parcel's buildings of footprint × ⌊height / 10 ft⌋ ([src/metrics.py](../../src/metrics.py)).

## Methods

| Method | Idea | Final flag | Notebook |
|---|---|---|---|
| 1. Floor-area agreement | Parcels whose LARIAC floor area matches assessor `SQFTmain` most closely leave the least room for unrecorded space. | Residual screen | [01](01_floor_area_assessor_method.ipynb) |
| 2. Structural survival and UDU similarity | Parcels that are structurally unchanged from 2014 to 2017, and whose characteristics lie far from known UDUs of the same use type. | Similarity | [02](02_structural_survival_method.ipynb) |
| 3. UDU propensity | XGBoost trained to separate known UDUs from matched unreported parcels; the lowest-scoring parcels are flagged. | Model named in `method3_final_model` | [03](03_udu_propensity_method.ipynb) |

[04](04_method2_vs_method3.ipynb) compares the methods and summarizes the findings.

## Columns

### Identity and summary

| Column | Type | Description |
|---|---|---|
| `AIN` | text | Assessor Identification Number. |
| `UseType` | text | 2016 assessor specific use type (`usecodedescchar2_2016` in `all_parcel_data`). |
| `LandValueBand` | integer 0–10 | Decile of 2016 land value per square foot of lot, within use type; 0 = no positive land value. Used by the Method 3 land-value model. |
| `KnownUDU` | flag | The AIN is in the current UDU report. |
| `methods_scoring` | integer 0–3 | Number of methods that scored the AIN. |
| `methods_flagging` | integer 0–3 | Number of the three final flags that are true. |
| `all_methods_potential_nonudu` | flag | True only when all three final flags are true. |

### Method 1: floor-area agreement ([notebook 01](01_floor_area_assessor_method.ipynb))

| Column | Description |
|---|---|
| `method1_potential_nonudu` | Final flag; equal to the residual screen. |
| `method1_residual_sqft` | LARIAC floor area − `SQFTmain`, in ft². Positive when LARIAC shows more area. |
| `method1_residual_score` | Within-use-type percentile rank of \|residual\|. **Low = close agreement.** |
| `method1_residual_potential_nonudu` | `method1_residual_score` ≤ 0.025 and the residual is positive. |
| `method1_ratio` | LARIAC floor area / `SQFTmain`. |
| `method1_ratio_score` | Within-use-type percentile rank of the ratio. **Low = the assessor records at least as much area as LARIAC shows.** |
| `method1_ratio_potential_nonudu` | `method1_ratio_score` ≤ 0.05. |

### Method 2: structural survival and UDU similarity ([notebook 02](02_structural_survival_method.ipynb))

| Column | Description |
|---|---|
| `method2_potential_nonudu` | Final flag: the parcel passes the structural screen, is not a known UDU, and `method2_udu_like` is false. |
| `method2_structural_screen` | All six checks below pass. |
| `method2_check_eligible_snapshot` | The parcel is eligible (see Conventions). |
| `method2_check_no_structural_change` | Footprints unchanged 2014–2017 (`change_type_1417_2017` = `no_change`, with no change in building count). |
| `method2_check_assessor_stable` | 2017 `SQFTmain` equals 2016. |
| `method2_check_lot_size` | Lot area within the use-type maximum (`TYPE_CONFIG` in notebook 02). |
| `method2_check_building_count` | Between 1 and the use-type maximum number of buildings. |
| `method2_check_floor_area_ratio` | LARIAC floor area / `SQFTmain` within the use-type bounds. |
| `method2_distance` | Mahalanobis distance to known UDUs of the same use type, over 15 structural, assessor and value features. Blank when the screen fails. |
| `method2_distance_score` | Share of known UDUs (leave-one-out) at or closer than this parcel. **High = less UDU-like.** Coarse where few known UDUs exist: single-family scores take 14 values. |
| `method2_udu_like` | Distance at or below the 90th percentile of known-UDU distances. |
| `method2_status` | `not_structurally_stable`, `udu_like`, `potential_nonudu` or `known_udu`. |

### Method 3: UDU propensity ([notebook 03](03_udu_propensity_method.ipynb))

Each of the three models has three columns:
- `_score`: the mean predicted UDU probability from balanced XGBoost bags. **High = more UDU-like.** Use it as a ranking, not a calibrated probability.
- `_rank`: the score's within-use-type percentile rank.
- `_potential_nonudu`: the model's flag.

The features cover assessor and LARIAC building metrics, parcel shape and open space, building-shape summaries, permit, housing-code, zoning and assessor history through 2017, and TCAC/HCD 2021 opportunity scores.

| Column | Description |
|---|---|
| `method3_potential_nonudu` | Final flag, from the model in `method3_final_model`. |
| `method3_final_model` | `baseline` (the default), `regularized` or `landvalue`. |
| `method3_baseline_score`, `_rank`, `_potential_nonudu` | **Tract-matched model.** Trained in the 293 census tracts that contain known UDUs, with each UDU matched to up to 20 unreported parcels of the same use type and tract; 143 features. Flag: `_rank` ≤ 0.05. The other ~75% of parcels receive extrapolated scores, and the ranking closely tracks land value. |
| `method3_regularized_score`, `_rank`, `_potential_nonudu` | As the baseline, with duplicate features removed (105 remain) and stronger XGBoost penalties. |
| `method3_landvalue_score`, `_rank`, `_potential_nonudu` | **Land-value-matched model.** Each UDU is matched on use type × `LandValueBand` across the whole city; cross-validation is grouped by 1-mile cells; the TCAC features are dropped. Every parcel is scored out of fold. Flag: `method3_landvalue_band_rank` ≤ 0.05. |
| `method3_landvalue_band_rank` | The land-value score's percentile rank within use type × `LandValueBand`. |

## Caveats

- **The methods mostly flag different parcels.** Only 61 AINs (0.011%) carry all three final flags; see the findings in [notebook 04](04_method2_vs_method3.ipynb).
- **Areas are not directly comparable across methods.** Method 2 and the Method 3 baseline disagree about which areas look least UDU-like, and the baseline's area pattern largely reflects land value.
- **Scores are relative to one run.** Ranks and flags depend on the parcels and known UDUs in that run, so compare them only within the same file.
