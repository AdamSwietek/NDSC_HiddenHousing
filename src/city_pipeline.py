"""Restartable, bounded-memory execution for large polygon AOIs."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from .io import (
    LARIAC_SOURCE_COVERAGE,
    build_assessor_cache,
    load_parcel_polygons,
)
from .outputs import OUTPUT_SCHEMA_VERSION, write_feature_outputs
from .pipeline import (
    DEFAULT_ASSESSOR_COLUMNS,
    ChangeJob,
    FeaturePipelineResult,
    SnapshotJob,
    run_feature_pipeline,
)


METRES_PER_MILE = 1609.344
METRES_PER_FOOT = 0.3048


@dataclass(frozen=True)
class ProcessingTile:
    tile_id: str
    column: int
    row: int
    core: object
    halo: object


@dataclass(frozen=True)
class TileGrid:
    crs: str
    origin_x: float
    origin_y: float
    tile_size_m: float
    max_column: int
    max_row: int


def load_polygon_aoi(path) -> gpd.GeoDataFrame:
    """Load, validate, and union an AOI dataset in WGS84."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    frame = gpd.read_file(path)
    if frame.crs is None:
        raise ValueError(f'AOI file has no CRS: {path}')
    present = pd.Series(
        ~pd.isna(frame.geometry.array), index=frame.index
    ) & ~frame.geometry.is_empty
    frame = frame.loc[present]
    if frame.empty:
        raise ValueError(f'AOI file has no usable geometry: {path}')
    geometry = frame.to_crs('EPSG:4326').geometry.make_valid().union_all()
    if geometry.is_empty or not geometry.is_valid:
        raise ValueError(f'AOI union is invalid: {path}')
    return gpd.GeoDataFrame({'aoi_name': [path.stem]}, geometry=[geometry], crs='EPSG:4326')


def make_processing_tiles(
    aoi,
    *,
    tile_size_miles=5.0,
    halo_feet=500.0,
    working_crs='EPSG:26911',
):
    """Create non-overlapping core tiles and buffered source-load halos."""
    if tile_size_miles <= 0 or halo_feet < 0:
        raise ValueError('tile size must be positive and halo must be non-negative')
    projected = aoi.to_crs(working_crs)
    city = projected.geometry.union_all()
    minx, miny, maxx, maxy = city.bounds
    size = float(tile_size_miles) * METRES_PER_MILE
    halo = float(halo_feet) * METRES_PER_FOOT
    max_column = max(0, int(np.ceil((maxx - minx) / size)) - 1)
    max_row = max(0, int(np.ceil((maxy - miny) / size)) - 1)
    grid = TileGrid(
        str(projected.crs), minx, miny, size, max_column, max_row
    )
    tiles = []
    for row in range(max_row + 1):
        for column in range(max_column + 1):
            cell = shapely.box(
                minx + column * size,
                miny + row * size,
                min(minx + (column + 1) * size, maxx),
                min(miny + (row + 1) * size, maxy),
            )
            core = city.intersection(cell)
            if core.is_empty or core.area == 0:
                continue
            tile_id = f'tile_r{row:03d}_c{column:03d}'
            tiles.append(ProcessingTile(tile_id, column, row, core, core.buffer(halo)))
    return tiles, grid


def run_tiled_feature_pipeline(
    aoi,
    *,
    snapshot_jobs,
    change_jobs,
    output_root,
    work_root,
    run_name,
    tile_size_miles=5.0,
    halo_feet=500.0,
    snapshot_crs='EPSG:2229',
    change_crs='EPSG:26911',
    assessor_columns=DEFAULT_ASSESSOR_COLUMNS,
    assessor_cache_path=None,
    alignment_sample_tiles=12,
    write_change_geometries=True,
    overwrite=False,
):
    """Run a resumable tiled pipeline and atomically publish merged outputs."""
    snapshots = [job if isinstance(job, SnapshotJob) else SnapshotJob(*job) for job in snapshot_jobs]
    changes = [job if isinstance(job, ChangeJob) else ChangeJob(*job) for job in change_jobs]
    if not snapshots and not changes:
        raise ValueError('Provide at least one snapshot or change job')
    _require_full_lariac_coverage(snapshots, changes)
    tiles, grid = make_processing_tiles(
        aoi,
        tile_size_miles=tile_size_miles,
        halo_feet=halo_feet,
        working_crs=change_crs,
    )
    if not tiles:
        raise ValueError('AOI produced no processing tiles')
    parcel_years = {
        job.parcel_year for job in snapshots
    } | {
        job.reference_parcel_year for job in changes
    }
    tiles, parcel_ains = _expand_tiles_to_owned_parcels(
        tiles,
        grid,
        aoi,
        parcel_years=parcel_years,
        assessor_parcel_years={job.parcel_year for job in snapshots},
        halo_feet=halo_feet,
    )

    work_path = Path(work_root) / run_name
    tile_root = work_path / 'tiles'
    tile_root.mkdir(parents=True, exist_ok=True)
    configuration = {
        'schema_version': OUTPUT_SCHEMA_VERSION,
        'run_name': run_name,
        'aoi_bounds_wgs84': [float(value) for value in aoi.total_bounds],
        'snapshot_jobs': [job.__dict__ for job in snapshots],
        'change_jobs': [job.__dict__ for job in changes],
        'tile_size_miles': float(tile_size_miles),
        'halo_feet': float(halo_feet),
        'snapshot_crs': snapshot_crs,
        'change_crs': change_crs,
        'assessor_columns': list(assessor_columns),
        'write_change_geometries': bool(write_change_geometries),
    }
    _validate_or_write_configuration(work_path / 'configuration.json', configuration)

    cache = None
    if snapshots:
        roll_years = sorted({job.lariac_year for job in snapshots})
        cache_label = '_'.join(str(year) for year in roll_years)
        cache = Path(
            assessor_cache_path
            or (work_path / f'assessor_{cache_label}.sqlite')
        )
        build_assessor_cache(
            cache,
            roll_years=roll_years,
            columns=assessor_columns,
            aoi=aoi,
            parcel_ains=parcel_ains,
        )

    alignments = _load_or_estimate_alignments(
        work_path / 'alignments.json',
        tiles,
        grid,
        aoi,
        changes,
        sample_count=alignment_sample_tiles,
        snapshot_crs=snapshot_crs,
        change_crs=change_crs,
    )

    city_projected = aoi.to_crs(grid.crs).geometry.union_all()
    for tile in tiles:
        tile_path = tile_root / tile.tile_id
        if (tile_path / 'manifest.json').exists():
            continue
        halo_aoi = gpd.GeoDataFrame(
            {'tile_id': [tile.tile_id]}, geometry=[tile.halo], crs=grid.crs
        ).to_crs('EPSG:4326')
        result = run_feature_pipeline(
            halo_aoi,
            snapshot_jobs=snapshots,
            change_jobs=changes,
            snapshot_crs=snapshot_crs,
            change_crs=change_crs,
            assessor_columns=assessor_columns,
            assessor_db_filepath=cache,
            change_alignments=alignments,
            retain_change_debug_geometries=False,
        )
        _retain_owned_outputs(result, tile, grid, city_projected)
        write_feature_outputs(
            result,
            tile_root,
            tile.tile_id,
            aoi_wgs84=halo_aoi.total_bounds,
            parameters={**configuration, 'tile_id': tile.tile_id},
            write_change_geometries=write_change_geometries,
        )

    return merge_tile_outputs(
        tile_root,
        Path(output_root),
        run_name,
        configuration=configuration,
        alignments=alignments,
        expected_tile_ids=[tile.tile_id for tile in tiles],
        overwrite=overwrite,
    )


def _expand_tiles_to_owned_parcels(
    tiles,
    grid,
    aoi,
    *,
    parcel_years,
    assessor_parcel_years,
    halo_feet,
):
    """Ensure each tile halo contains the full extent of every owned parcel."""
    city = aoi.to_crs(grid.crs).geometry.union_all()
    halo_m = float(halo_feet) * METRES_PER_FOOT
    bounds_by_tile = {tile.tile_id: [] for tile in tiles}
    assessor_ains = []
    tile_lookup = {(tile.column, tile.row): tile.tile_id for tile in tiles}
    for year in sorted(parcel_years):
        parcels = load_parcel_polygons(
            year,
            aoi,
            columns=['ASSETID', 'PIND', 'PIN'],
            target_crs=grid.crs,
            include_missing_ain=True,
        )
        if year in assessor_parcel_years:
            assessor_ains.extend(parcels['AIN'].dropna().tolist())
        clipped = parcels.geometry.intersection(city)
        present = pd.Series(
            ~pd.isna(clipped.array), index=clipped.index
        ) & ~clipped.is_empty
        points = clipped.representative_point()
        columns = np.floor((points.x - grid.origin_x) / grid.tile_size_m).clip(
            0, grid.max_column
        ).astype('Int64')
        rows = np.floor((points.y - grid.origin_y) / grid.tile_size_m).clip(
            0, grid.max_row
        ).astype('Int64')
        owners = pd.DataFrame({'column': columns, 'row': rows}, index=parcels.index)
        owners = owners.loc[present]
        for (column, row), indices in owners.groupby(['column', 'row']).groups.items():
            tile_id = tile_lookup.get((int(column), int(row)))
            if tile_id is None:
                continue
            bounds_by_tile[tile_id].append(
                parcels.loc[list(indices)].total_bounds
            )

    expanded = []
    for tile in tiles:
        halo = tile.halo
        for minx, miny, maxx, maxy in bounds_by_tile[tile.tile_id]:
            parcel_envelope = shapely.box(minx, miny, maxx, maxy).buffer(halo_m)
            halo = halo.union(parcel_envelope)
        expanded.append(
            ProcessingTile(tile.tile_id, tile.column, tile.row, tile.core, halo)
        )
    return expanded, assessor_ains


def _load_or_estimate_alignments(
    path,
    tiles,
    grid,
    aoi,
    change_jobs,
    *,
    sample_count,
    snapshot_crs,
    change_crs,
):
    if not change_jobs:
        return {}
    path = Path(path)
    if path.exists():
        stored = json.loads(path.read_text())
        return {
            tuple(int(value) for value in key.split('_')): value
            for key, value in stored.items()
        }
    count = min(max(1, int(sample_count)), len(tiles))
    positions = sorted(set(np.linspace(0, len(tiles) - 1, count, dtype=int)))
    estimates = {(job.start_year, job.end_year): [] for job in change_jobs}
    for position in positions:
        tile = tiles[position]
        sample_aoi = gpd.GeoDataFrame(
            geometry=[tile.halo], crs=grid.crs
        ).to_crs('EPSG:4326')
        result = run_feature_pipeline(
            sample_aoi,
            change_jobs=change_jobs,
            snapshot_crs=snapshot_crs,
            change_crs=change_crs,
            retain_change_debug_geometries=False,
        )
        for pair, output in result.change_results.items():
            diagnostic = output['diagnostics']
            if diagnostic.get('alignment_applied'):
                estimates[pair].append(diagnostic)
    alignments = {}
    for pair, records in estimates.items():
        if records:
            alignments[pair] = {
                'shift_dx_m': float(np.median([r['shift_dx_m'] for r in records])),
                'shift_dy_m': float(np.median([r['shift_dy_m'] for r in records])),
                'alignment_candidates': int(sum(r['alignment_candidates'] for r in records)),
                'alignment_used': int(sum(r['alignment_used'] for r in records)),
                'alignment_residual_median_m': float(np.median([
                    r['alignment_residual_median_m'] for r in records
                ])),
            }
        else:
            alignments[pair] = {'shift_dx_m': 0.0, 'shift_dy_m': 0.0}
    serializable = {f'{pair[0]}_{pair[1]}': value for pair, value in alignments.items()}
    path.write_text(json.dumps(serializable, indent=2, sort_keys=True) + '\n')
    return alignments


def _retain_owned_outputs(result, tile, grid, city_projected):
    for year, frame in result.parcel_metrics.items():
        owned = _owned_keys(frame, 'site_id', tile, grid, city_projected)
        result.parcel_metrics[year] = _filter_frame(frame, 'site_id', owned)
        buildings = result.building_metrics[year]
        matched = buildings['site_id'].isin(owned)
        unmatched = buildings['site_id'].isna() & _owned_geometry_mask(
            buildings, tile, grid, city_projected
        )
        owned_buildings = buildings.loc[matched | unmatched].copy()
        owned_buildings.attrs.update(buildings.attrs)
        _retain_owned_repairs(owned_buildings, owned)
        halo_diagnostics = buildings.attrs.get('matching_diagnostics', {})
        owned_diagnostics = _owned_snapshot_diagnostics(
            owned_buildings, result.parcel_metrics[year]
        )
        owned_buildings.attrs['halo_matching_diagnostics'] = halo_diagnostics
        owned_buildings.attrs['matching_diagnostics'] = owned_diagnostics
        result.parcel_metrics[year].attrs['halo_matching_diagnostics'] = (
            halo_diagnostics
        )
        result.parcel_metrics[year].attrs['matching_diagnostics'] = owned_diagnostics
        result.building_metrics[year] = owned_buildings
        bridge = result.assessor_ain_metrics[year]
        result.assessor_ain_metrics[year] = bridge.loc[
            bridge['site_id'].isin(owned)
        ].copy()
        snapshot_diagnostics = result.diagnostics['snapshots'][year]
        snapshot_diagnostics['halo_matching'] = snapshot_diagnostics['matching']
        snapshot_diagnostics['matching'] = owned_diagnostics

    for pair, output in result.change_results.items():
        owned = _owned_keys(
            output['metrics'], 'reference_site_id', tile, grid, city_projected
        )
        output['metrics'] = _filter_frame(
            output['metrics'], 'reference_site_id', owned
        )
        for key in ('added', 'removed'):
            if key in output:
                output[key] = _filter_frame(output[key], 'reference_site_id', owned)
        for year, footprints in output['footprints'].items():
            owned_footprints = _filter_frame(
                footprints, 'reference_site_id', owned
            )
            _retain_owned_repairs(owned_footprints, owned)
            output['footprints'][year] = owned_footprints
    result.diagnostics['rows'] = result.row_summary()


def _owned_keys(frame, key, tile, grid, city_projected):
    mask = _owned_geometry_mask(frame, tile, grid, city_projected)
    return set(frame.loc[mask, key].dropna())


def _owned_geometry_mask(frame, tile, grid, city_projected):
    if frame.empty:
        return pd.Series(False, index=frame.index)
    projected = frame.to_crs(grid.crs)
    clipped = projected.geometry.intersection(city_projected)
    present = pd.Series(
        ~pd.isna(clipped.array), index=clipped.index
    ) & ~clipped.is_empty
    points = clipped.representative_point()
    columns = np.floor((points.x - grid.origin_x) / grid.tile_size_m).clip(
        0, grid.max_column
    )
    rows = np.floor((points.y - grid.origin_y) / grid.tile_size_m).clip(
        0, grid.max_row
    )
    return present & columns.eq(tile.column) & rows.eq(tile.row)


def _filter_frame(frame, key, values):
    result = frame.loc[frame[key].isin(values)].copy()
    result.attrs.update(frame.attrs)
    return result


def _retain_owned_repairs(frame, owned_sites):
    records = frame.attrs.get('boundary_repair_records')
    if records is None:
        return
    frame.attrs['boundary_repair_records'] = [
        record for record in records
        if record.get('dominant_site_id') in owned_sites
        or (
            record.get('dominant_site_id') is None
            and record.get('dominant_AIN') is not None
            and f"ain:{record['dominant_AIN']}" in owned_sites
        )
    ]


def _owned_snapshot_diagnostics(buildings, parcels):
    present_buildings = pd.Series(
        ~pd.isna(buildings.geometry.array), index=buildings.index
    ) & ~buildings.geometry.is_empty
    present_parcels = pd.Series(
        ~pd.isna(parcels.geometry.array), index=parcels.index
    ) & ~parcels.geometry.is_empty
    matched = buildings['site_id'].notna()
    methods = buildings.get(
        'match.method', pd.Series(pd.NA, index=buildings.index)
    )
    return {
        'input_buildings': int(len(buildings)),
        'valid_buildings': int(present_buildings.sum()),
        'matched_buildings': int(matched.sum()),
        'unmatched_buildings': int((present_buildings & ~matched).sum()),
        'invalid_or_empty_buildings': int((~present_buildings).sum()),
        'representative_point_matches': int(
            methods.eq('representative_point').sum()
        ),
        'maximum_overlap_matches': int(methods.eq('maximum_overlap').sum()),
        'input_parcels': int(len(parcels)),
        'invalid_or_empty_parcels': int((~present_parcels).sum()),
    }


def merge_tile_outputs(
    tile_root,
    output_root,
    run_name,
    *,
    configuration,
    alignments,
    expected_tile_ids,
    overwrite=False,
):
    """Validate, merge, and atomically publish completed tile outputs."""
    tile_root = Path(tile_root)
    tile_dirs = [tile_root / tile_id for tile_id in expected_tile_ids]
    missing = [path.name for path in tile_dirs if not (path / 'manifest.json').exists()]
    if missing:
        raise RuntimeError(f'Missing {len(missing)} tile outputs; first: {missing[:5]}')
    filenames = sorted({
        entry['file']
        for directory in tile_dirs
        for entry in json.loads((directory / 'manifest.json').read_text())['outputs']
    })
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / run_name
    if target.exists() and not overwrite:
        raise FileExistsError(
            f'Output run already exists: {target}; pass --overwrite to replace it'
        )
    staging = Path(tempfile.mkdtemp(prefix=f'.{run_name}.merge-', dir=output_root))
    validations = {}
    output_entries = []
    try:
        for filename in filenames:
            paths = [directory / filename for directory in tile_dirs if (directory / filename).exists()]
            spatial = not filename.startswith(
                ('assessor_ain_metrics_', 'boundary_repairs_')
            )
            frames = [gpd.read_parquet(path) if spatial else pd.read_parquet(path) for path in paths]
            frames = [_normalize_output_identifier_types(frame, filename) for frame in frames]
            merged = pd.concat(frames, ignore_index=True)
            if spatial:
                merged = gpd.GeoDataFrame(merged, geometry=frames[0].geometry.name, crs=frames[0].crs)
            if filename.startswith('building_metrics_'):
                merged = _disambiguate_reused_structure_ids(merged)
            key = _output_key(filename)
            duplicates = int(merged.duplicated(key).sum()) if not merged.empty else 0
            if duplicates and not filename.startswith('boundary_repairs_'):
                raise RuntimeError(f'{filename} contains {duplicates} duplicate {key} values')
            if duplicates:
                merged = merged.drop_duplicates(key).reset_index(drop=True)
            destination = staging / filename
            merged.to_parquet(destination, index=False, compression='zstd')
            validations[filename] = {
                'tile_rows': int(sum(len(frame) for frame in frames)),
                'merged_rows': int(len(merged)),
                'duplicate_keys': duplicates,
                'key': list(key) if isinstance(key, tuple) else key,
            }
            output_entries.append({
                'file': filename,
                'rows': int(len(merged)),
                'size_bytes': int(destination.stat().st_size),
                'crs': str(merged.crs) if spatial else None,
                'geometry_column': merged.geometry.name if spatial else None,
            })
        relationships = _validate_merged_relationships(staging)
        manifest = {
            'schema_version': OUTPUT_SCHEMA_VERSION,
            'run_name': run_name,
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'execution': 'tiled',
            'tile_count': len(tile_dirs),
            'parameters': configuration,
            'alignments': {f'{a}_{b}': value for (a, b), value in alignments.items()},
            'validation': validations,
            'relationship_validation': relationships,
            'outputs': output_entries,
        }
        (staging / 'manifest.json').write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str) + '\n'
        )
        _publish(staging, target, overwrite)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return target


def _require_full_lariac_coverage(snapshot_jobs, change_jobs):
    years = {job.lariac_year for job in snapshot_jobs}
    years.update(
        year for job in change_jobs for year in (job.start_year, job.end_year)
    )
    partial = {
        year: LARIAC_SOURCE_COVERAGE.get(year, 'unknown')
        for year in sorted(years)
        if LARIAC_SOURCE_COVERAGE.get(year) != 'countywide'
    }
    if partial:
        descriptions = ', '.join(
            f'{year} ({coverage})' for year, coverage in partial.items()
        )
        raise RuntimeError(
            'Citywide processing requires full-coverage LARIAC sources; '
            f'not full coverage: {descriptions}. Register the complete vintage '
            'and update LARIAC_SOURCE_COVERAGE before running.'
        )


def _validate_merged_relationships(directory):
    """Check that child output identifiers resolve to their parent outputs."""
    directory = Path(directory)
    checks = {}
    for parcel_path in directory.glob('parcel_metrics_*.parquet'):
        year = parcel_path.stem.rsplit('_', 1)[1]
        parcel_sites = set(pd.read_parquet(parcel_path, columns=['site_id'])['site_id'])
        for prefix, child_key in (
            ('building_metrics', 'site_id'),
            ('assessor_ain_metrics', 'site_id'),
        ):
            child_path = directory / f'{prefix}_{year}.parquet'
            if not child_path.exists():
                continue
            child = pd.read_parquet(child_path, columns=[child_key])[child_key].dropna()
            missing = set(child).difference(parcel_sites)
            if missing:
                raise RuntimeError(
                    f'{child_path.name} contains {len(missing)} unknown site_id values'
                )
            checks[f'{child_path.name}.site_ids_resolve'] = True

    for change_path in directory.glob('change_metrics_*.parquet'):
        pair = change_path.stem.removeprefix('change_metrics_')
        sites = set(pd.read_parquet(
            change_path, columns=['reference_site_id']
        )['reference_site_id'])
        for prefix in ('added', 'removed'):
            geometry_path = directory / f'{prefix}_{pair}.parquet'
            if not geometry_path.exists():
                continue
            child = pd.read_parquet(
                geometry_path, columns=['reference_site_id']
            )['reference_site_id'].dropna()
            missing = set(child).difference(sites)
            if missing:
                raise RuntimeError(
                    f'{geometry_path.name} contains {len(missing)} unknown sites'
                )
            checks[f'{geometry_path.name}.site_ids_resolve'] = True
    return checks


def _output_key(filename):
    if filename.startswith('building_metrics_'):
        return 'structure_id'
    if filename.startswith('parcel_metrics_'):
        return 'site_id'
    if filename.startswith('assessor_ain_metrics_'):
        return 'AIN'
    if filename.startswith('change_metrics_') or filename.startswith(('added_', 'removed_')):
        return 'reference_site_id'
    if filename.startswith('boundary_repairs_'):
        return ('fragment_id', 'dominant_id')
    raise ValueError(f'Unknown tile output: {filename}')


def _normalize_output_identifier_types(frame, filename):
    """Normalize IDs across checkpoints written before and after schema fixes."""
    result = frame.copy()
    if filename.startswith('building_metrics_') and 'structure_id' in result:
        result['structure_id'] = result['structure_id'].astype('string')
    elif filename.startswith('boundary_repairs_'):
        for column in ('fragment_id', 'dominant_id'):
            if column in result:
                result[column] = result[column].astype('string')
    return result


def _disambiguate_reused_structure_ids(frame):
    """Resolve source IDs reused by distinct structures in separate tiles."""
    result = frame.copy()
    reused = result['structure_id'].duplicated(keep=False)
    if not reused.any():
        return result
    result.loc[reused, 'structure_id'] = [
        f'{value}~{_output_geometry_fingerprint(geometry)}'
        for value, geometry in zip(
            result.loc[reused, 'structure_id'],
            result.loc[reused, result.geometry.name],
        )
    ]
    return result


def _output_geometry_fingerprint(geometry):
    if geometry is None or geometry.is_empty:
        return 'missing'
    canonical = shapely.normalize(geometry)
    wkb = shapely.to_wkb(canonical, byte_order=1, output_dimension=2)
    return hashlib.blake2b(wkb, digest_size=8).hexdigest()


def _validate_or_write_configuration(path, configuration):
    path = Path(path)
    rendered = json.dumps(configuration, indent=2, sort_keys=True) + '\n'
    if path.exists() and json.loads(path.read_text()) != configuration:
        raise RuntimeError(
            f'Existing tiled run has different parameters: {path.parent}'
        )
    if not path.exists():
        path.write_text(rendered)


def _publish(staging, target, overwrite):
    if not target.exists():
        staging.rename(target)
        return
    if not overwrite:
        raise FileExistsError(target)
    backup = target.with_name(f'.{target.name}.backup')
    if backup.exists():
        raise FileExistsError(f'Stale output backup exists: {backup}')
    target.rename(backup)
    try:
        staging.rename(target)
    except Exception:
        backup.rename(target)
        raise
    shutil.rmtree(backup)
