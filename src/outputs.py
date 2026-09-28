"""Validated, atomic persistence for feature-pipeline outputs."""

from __future__ import annotations

import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd

from .pipeline import FeaturePipelineResult


OUTPUT_SCHEMA_VERSION = 4
BOUNDARY_REPAIR_COLUMNS = [
    'lariac_year',
    'fragment_id',
    'dominant_id',
    'fragment_AIN',
    'dominant_AIN',
    'dominant_site_id',
    'fragment_area_ft2',
    'shared_edge_ft',
    'parcel_boundary_coverage',
    'sliver_buffer_ft',
    'opened_retained_fraction',
    'fragment_parcel_index',
    'dominant_parcel_index',
]


def write_feature_outputs(
    result: FeaturePipelineResult,
    output_root,
    run_name: str,
    *,
    aoi_wgs84,
    parameters=None,
    overwrite=False,
    write_change_geometries=False,
) -> Path:
    """Write a complete pipeline run and return its published directory.

    Files are first written to a temporary sibling directory. The run becomes
    visible at ``output_root/run_name`` only after every validation and write
    succeeds. Existing runs are protected unless ``overwrite`` is true.
    """
    if not isinstance(result, FeaturePipelineResult):
        raise TypeError('result must be a FeaturePipelineResult')
    _validate_run_name(run_name)
    _validate_outputs(result)

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / run_name
    if target.exists() and not overwrite:
        raise FileExistsError(
            f'Output run already exists: {target}; pass overwrite=True to replace it'
        )

    temporary = Path(tempfile.mkdtemp(prefix=f'.{run_name}.tmp-', dir=root))
    try:
        output_entries = []
        for year, frame in sorted(result.building_metrics.items()):
            output_entries.append(
                _write_spatial(
                    frame,
                    temporary / f'building_metrics_{year}.parquet',
                    'building_metrics',
                )
            )
        for year, frame in sorted(result.parcel_metrics.items()):
            output_entries.append(
                _write_spatial(
                    frame,
                    temporary / f'parcel_metrics_{year}.parquet',
                    'parcel_metrics',
                )
            )
        for year, frame in sorted(result.assessor_ain_metrics.items()):
            path = temporary / f'assessor_ain_metrics_{year}.parquet'
            frame.to_parquet(path, index=False, compression='zstd')
            output_entries.append(
                _file_entry(path, 'assessor_ain_metrics', len(frame), None, None)
            )
        for (start, end), change_result in sorted(result.change_results.items()):
            output_entries.append(
                _write_spatial(
                    change_result['metrics'],
                    temporary / f'change_metrics_{start}_{end}.parquet',
                    'change_metrics',
                )
            )
            if write_change_geometries:
                for key in ('added', 'removed'):
                    output_entries.append(
                        _write_spatial(
                            change_result[key],
                            temporary / f'{key}_{start}_{end}.parquet',
                            key,
                        )
                    )

        for year, repairs in sorted(_boundary_repairs_by_year(result).items()):
            path = temporary / f'boundary_repairs_{year}.parquet'
            repairs.to_parquet(path, index=False, compression='zstd')
            output_entries.append(
                _file_entry(path, 'boundary_repairs', len(repairs), None, None)
            )

        manifest = {
            'schema_version': OUTPUT_SCHEMA_VERSION,
            'run_name': run_name,
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'aoi_wgs84': [float(value) for value in aoi_wgs84],
            'parameters': parameters or {},
            'rows': result.row_summary(),
            'diagnostics': result.diagnostics,
            'outputs': output_entries,
        }
        manifest_path = temporary / 'manifest.json'
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=_json_default)
            + '\n'
        )

        _publish_directory(temporary, target, overwrite)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return target


def _validate_run_name(run_name):
    if not run_name or Path(run_name).name != run_name or run_name in {'.', '..'}:
        raise ValueError('run_name must be one non-empty directory name')


def _validate_outputs(result):
    for year, frame in result.building_metrics.items():
        _validate_spatial(frame, f'building metrics {year}')
        key = 'structure_id' if 'structure_id' in frame else 'building_id'
        if key not in frame or frame[key].duplicated().any():
            raise ValueError(f'Building output {year} must have unique {key}')
    for year, frame in result.parcel_metrics.items():
        _validate_spatial(frame, f'parcel metrics {year}')
        if 'site_id' not in frame or frame['site_id'].duplicated().any():
            raise ValueError(
                f'Parcel output {year} must have unique site_id values'
            )
    for year, frame in result.assessor_ain_metrics.items():
        if 'AIN' not in frame or frame['AIN'].duplicated().any():
            raise ValueError(
                f'Assessor AIN output {year} must have unique AIN values'
            )
    for pair, change_result in result.change_results.items():
        if 'metrics' not in change_result:
            raise ValueError(f'Change output {pair} is missing metrics')
        frame = change_result['metrics']
        _validate_spatial(frame, f'change metrics {pair}')
        if (
            'reference_site_id' not in frame
            or frame['reference_site_id'].duplicated().any()
        ):
            raise ValueError(
                f'Change output {pair} must have unique reference_site_id values'
            )


def _validate_spatial(frame, label):
    if not isinstance(frame, gpd.GeoDataFrame):
        raise TypeError(f'{label} must be a GeoDataFrame')
    if frame.crs is None:
        raise ValueError(f'{label} is missing a CRS')
    if frame.geometry.name not in frame:
        raise ValueError(f'{label} is missing its active geometry column')


def _write_spatial(frame, path, dataset_type):
    safe, json_columns = _parquet_safe_copy(frame)
    safe.to_parquet(path, index=False, compression='zstd')
    return _file_entry(
        path,
        dataset_type,
        len(safe),
        safe.crs.to_string(),
        safe.geometry.name,
        json_columns,
    )


def _parquet_safe_copy(frame):
    safe = frame.copy()
    json_columns = []
    for column in safe.columns:
        if column == safe.geometry.name or safe[column].dtype != 'object':
            continue
        values = safe[column].dropna()
        if values.map(lambda value: isinstance(value, (dict, list, tuple, set))).any():
            safe[column] = safe[column].map(
                lambda value: (
                    json.dumps(value, default=_json_default)
                    if isinstance(value, (dict, list, tuple, set))
                    else value
                )
            )
            json_columns.append(column)
    return safe, json_columns


def _boundary_repairs_by_year(result):
    records_by_year = {}
    for year, frame in result.building_metrics.items():
        records_by_year[int(year)] = frame.attrs.get(
            'boundary_repair_records', []
        )
    for change_result in result.change_results.values():
        for year, frame in change_result['footprints'].items():
            records_by_year.setdefault(
                int(year), frame.attrs.get('boundary_repair_records', [])
            )

    outputs = {}
    for year, records in records_by_year.items():
        frame = pd.DataFrame(records)
        if frame.empty:
            frame = pd.DataFrame(columns=BOUNDARY_REPAIR_COLUMNS[1:])
        frame.insert(0, 'lariac_year', int(year))
        for column in BOUNDARY_REPAIR_COLUMNS:
            if column not in frame:
                frame[column] = pd.NA
        outputs[year] = frame[BOUNDARY_REPAIR_COLUMNS].drop_duplicates(
            ['fragment_id', 'dominant_id']
        ).reset_index(drop=True)
    return outputs


def _file_entry(
    path,
    dataset_type,
    rows,
    crs,
    geometry_column,
    json_encoded_columns=None,
):
    return {
        'file': path.name,
        'dataset_type': dataset_type,
        'rows': int(rows),
        'crs': crs,
        'geometry_column': geometry_column,
        'json_encoded_columns': json_encoded_columns or [],
        'size_bytes': int(path.stat().st_size),
    }


def _publish_directory(temporary, target, overwrite):
    if not target.exists():
        temporary.rename(target)
        return
    if not overwrite:
        raise FileExistsError(target)
    backup = target.with_name(f'.{target.name}.backup')
    if backup.exists():
        raise FileExistsError(f'Stale output backup exists: {backup}')
    target.rename(backup)
    try:
        temporary.rename(target)
    except Exception:
        backup.rename(target)
        raise
    shutil.rmtree(backup)


def _json_default(value):
    if hasattr(value, 'item'):
        return value.item()
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)
