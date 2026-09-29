"""Parcel-level building-footprint change metrics across LARIAC epochs."""

from __future__ import annotations

from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.errors import GEOSException

try:
    from .lariac_cleaning import repair_boundary_fragments
    from .matching import align_matching_crs, match_buildings_to_parcels
    from .sites import add_parcel_site_ids
except ImportError:  # Support importing modules directly from the src directory.
    from lariac_cleaning import repair_boundary_fragments
    from matching import align_matching_crs, match_buildings_to_parcels
    from sites import add_parcel_site_ids


DEFAULT_CHANGE_CRS = 'EPSG:26911'
DEFAULT_SLIVER_BUFFER_M = 0.4
DEFAULT_MIN_CHANGE_AREA_M2 = 2.0
DEFAULT_RELATIVE_CHANGE = 0.02
TOPOLOGY_GRID_SIZE_M = 1e-6


def prepare_epoch_footprints(
    buildings: gpd.GeoDataFrame,
    reference_parcels: gpd.GeoDataFrame,
    year: int,
    *,
    target_crs=DEFAULT_CHANGE_CRS,
    split_necks_m: Optional[float] = None,
    repair_boundary_artifacts=True,
    repair_options=None,
) -> gpd.GeoDataFrame:
    """Assign buildings to a stable parcel geography and dissolve by site.

    Source-level parcel-boundary fragments are repaired before assignment
    unless ``repair_boundary_artifacts`` is false. The returned geometry is
    named ``footprint`` and has one row per ``reference_site_id``. Canonical
    ``reference_AIN`` is retained when available. Assignment
    diagnostics are stored in ``result.attrs``. ``split_necks_m`` optionally
    ports the notebook's erode/dilate cleanup, but is disabled by default
    because it changes the source footprint geometry.
    """
    year = int(year)
    if not isinstance(buildings, gpd.GeoDataFrame):
        raise TypeError('buildings must be a GeoDataFrame')
    parcels = _prepare_reference_parcels(reference_parcels)
    buildings, parcels = align_matching_crs(buildings, parcels, target_crs)
    if buildings.crs is None or buildings.crs.is_geographic:
        raise ValueError('Change metrics require a projected metric CRS')

    buildings = buildings.copy().reset_index(drop=True)
    present = _present_geometry(buildings.geometry)
    buildings = buildings.loc[present].copy().reset_index(drop=True)
    buildings.geometry = buildings.geometry.make_valid()
    polygonal = buildings.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])
    buildings = buildings.loc[polygonal].copy().reset_index(drop=True)
    if repair_boundary_artifacts:
        buildings = repair_boundary_fragments(
            buildings, parcels, **(repair_options or {})
        )
    repair_diagnostics = buildings.attrs.get('boundary_repair_diagnostics')
    repair_records = buildings.attrs.get('boundary_repair_records', [])
    if split_necks_m is not None:
        buildings = split_thin_connections(buildings, split_necks_m)

    buildings['_building_id'] = np.arange(len(buildings), dtype=np.int64)
    parcels['_parcel_id'] = np.arange(len(parcels), dtype=np.int64)
    matches = match_buildings_to_parcels(buildings, parcels)
    assigned = buildings.merge(
        matches, on='_building_id', how='inner', validate='one_to_one'
    )
    assigned = assigned.merge(
        parcels[['_parcel_id', 'site_id', 'AIN']],
        on='_parcel_id',
        how='left',
        validate='many_to_one',
    ).rename(columns={
        'site_id': 'reference_site_id',
        'AIN': 'reference_AIN',
    })
    if repair_records and 'structure_id' in assigned:
        structure_sites = assigned.set_index('structure_id')['reference_site_id']
        repair_records = [
            {
                **record,
                'dominant_site_id': structure_sites.get(record['dominant_id']),
            }
            for record in repair_records
        ]

    if assigned.empty:
        result = gpd.GeoDataFrame(
            columns=[
                'reference_site_id', 'reference_AIN', 'year', 'footprint_area_m2',
                'building_count', 'footprint',
            ],
            geometry='footprint',
            crs=buildings.crs,
        )
    else:
        assigned = gpd.GeoDataFrame(
            assigned, geometry=buildings.geometry.name, crs=buildings.crs
        )
        counts = assigned.groupby('reference_site_id').size()
        site_labels = assigned[
            ['reference_site_id', 'reference_AIN']
        ].drop_duplicates('reference_site_id')
        dissolved = assigned[
            ['reference_site_id', assigned.geometry.name]
        ].dissolve(
            by='reference_site_id', as_index=False
        ).merge(
            site_labels, on='reference_site_id', how='left', validate='one_to_one'
        )
        dissolved = dissolved.rename_geometry('footprint')
        dissolved['year'] = year
        dissolved['footprint_area_m2'] = dissolved.footprint.area
        dissolved['building_count'] = (
            dissolved['reference_site_id'].map(counts).astype('int64')
        )
        result = dissolved[
            [
                'reference_site_id', 'reference_AIN', 'year', 'footprint_area_m2',
                'building_count', 'footprint',
            ]
        ]

    matched_count = matches['_building_id'].nunique()
    result.attrs['assignment_diagnostics'] = {
        'year': year,
        'input_buildings': int(len(buildings)),
        'matched_buildings': int(matched_count),
        'unmatched_buildings': int(len(buildings) - matched_count),
        'reference_sites': int(len(parcels)),
        'reference_parcels': int(len(parcels)),
    }
    if repair_diagnostics is not None:
        result.attrs['boundary_repair_diagnostics'] = repair_diagnostics
        result.attrs['boundary_repair_records'] = repair_records
    result.attrs['year'] = year
    return result


def split_thin_connections(
    buildings: gpd.GeoDataFrame,
    threshold_m: float = 1.0,
) -> gpd.GeoDataFrame:
    """Sever connections narrower than about twice ``threshold_m``.

    This is the notebook's optional erode/dilate operation. Mitre joins retain
    rectangular corners; components smaller than the erosion kernel are dropped.
    """
    if threshold_m <= 0:
        raise ValueError('threshold_m must be positive')
    rows = []
    minimum_area = np.pi * threshold_m**2
    for _, row in buildings.iterrows():
        geometry = row.geometry
        eroded = geometry.buffer(-threshold_m, join_style=2)
        if eroded.is_empty:
            rows.append(row)
            continue
        restored = eroded.buffer(threshold_m, join_style=2)
        components = (
            list(restored.geoms)
            if restored.geom_type == 'MultiPolygon'
            else [restored]
        )
        for component in components:
            if component.area >= minimum_area:
                output = row.copy()
                output.geometry = component
                rows.append(output)
    return gpd.GeoDataFrame(rows, geometry=buildings.geometry.name, crs=buildings.crs)


def estimate_epoch_shift(
    start: gpd.GeoDataFrame,
    end: gpd.GeoDataFrame,
    *,
    min_alignment_parcels: int = 5,
    area_ratio_tolerance: float = 0.10,
) -> dict:
    """Estimate robust centroid translation from likely-stable parcels."""
    _validate_prepared_footprints(start, 'start')
    _validate_prepared_footprints(end, 'end')
    if start.crs != end.crs:
        raise ValueError('Prepared epoch CRS values must match')

    paired = start.merge(
        end,
        on='reference_site_id',
        suffixes=('_start', '_end'),
        validate='one_to_one',
    )
    n_common = len(paired)
    if paired.empty:
        return _empty_alignment(n_common)

    maximum_area = paired[
        ['footprint_area_m2_start', 'footprint_area_m2_end']
    ].max(axis=1)
    relative_area_delta = (
        (paired['footprint_area_m2_end'] - paired['footprint_area_m2_start'])
        .abs()
        .divide(maximum_area.where(maximum_area > 0))
    )
    candidates = paired[
        paired['building_count_start'].eq(paired['building_count_end'])
        & relative_area_delta.le(area_ratio_tolerance)
    ].copy()
    if len(candidates) < min_alignment_parcels:
        return _empty_alignment(n_common, len(candidates))

    start_centroids = gpd.GeoSeries(
        candidates['footprint_start'], crs=start.crs
    ).centroid
    end_centroids = gpd.GeoSeries(
        candidates['footprint_end'], crs=end.crs
    ).centroid
    dx = end_centroids.x.to_numpy() - start_centroids.x.to_numpy()
    dy = end_centroids.y.to_numpy() - start_centroids.y.to_numpy()
    median_dx = float(np.median(dx))
    median_dy = float(np.median(dy))
    residual = np.hypot(dx - median_dx, dy - median_dy)
    residual_median = float(np.median(residual))
    mad = float(np.median(np.abs(residual - residual_median)))
    cutoff = residual_median + 3.0 * max(mad, 0.05)
    keep = residual <= cutoff
    if int(keep.sum()) >= min_alignment_parcels:
        median_dx = float(np.median(dx[keep]))
        median_dy = float(np.median(dy[keep]))
        residual = np.hypot(dx[keep] - median_dx, dy[keep] - median_dy)

    return {
        'shift_dx_m': median_dx,
        'shift_dy_m': median_dy,
        'common_sites': int(n_common),
        'common_parcels': int(n_common),
        'alignment_candidates': int(len(candidates)),
        'alignment_used': int(keep.sum()),
        'alignment_residual_median_m': float(np.median(residual)),
        'alignment_applied': True,
    }


def apply_epoch_shift(
    footprints: gpd.GeoDataFrame,
    dx_m: float,
    dy_m: float,
) -> gpd.GeoDataFrame:
    """Translate the later epoch by ``(-dx_m, -dy_m)`` into the start frame."""
    shifted = footprints.copy()
    if abs(dx_m) < 1e-9 and abs(dy_m) < 1e-9:
        return shifted
    shifted['footprint'] = shifted.geometry.translate(
        xoff=-float(dx_m), yoff=-float(dy_m)
    )
    shifted = shifted.set_geometry('footprint')
    shifted['footprint_area_m2'] = shifted.geometry.area
    return shifted


def remove_slivers(geometry, buffer_m: float = DEFAULT_SLIVER_BUFFER_M):
    """Apply morphological opening and preserve mitred building corners."""
    if geometry is None or geometry.is_empty:
        return geometry
    if buffer_m < 0:
        raise ValueError('buffer_m must be non-negative')
    if buffer_m == 0:
        return geometry
    cleaned = geometry.buffer(-buffer_m, join_style=2)
    if cleaned.is_empty:
        return cleaned
    return cleaned.buffer(buffer_m, join_style=2)


def calculate_change_metrics(
    start: gpd.GeoDataFrame,
    end: gpd.GeoDataFrame,
    reference_parcels: gpd.GeoDataFrame,
    *,
    align=True,
    sliver_buffer_m: float = DEFAULT_SLIVER_BUFFER_M,
    min_change_area_m2: float = DEFAULT_MIN_CHANGE_AREA_M2,
    relative_change_threshold: float = DEFAULT_RELATIVE_CHANGE,
    min_alignment_parcels: int = 5,
    area_ratio_tolerance: float = 0.10,
    store_geometries=False,
    alignment_override=None,
) -> dict:
    """Compare prepared epoch footprints and classify parcel-level changes."""
    _validate_prepared_footprints(start, 'start')
    _validate_prepared_footprints(end, 'end')
    if start.crs != end.crs:
        raise ValueError('Prepared epoch CRS values must match')
    if (
        sliver_buffer_m < 0
        or min_change_area_m2 < 0
        or relative_change_threshold < 0
    ):
        raise ValueError('Change thresholds must be non-negative')

    start_year = _single_year(start, 'start')
    end_year = _single_year(end, 'end')
    if end_year <= start_year:
        raise ValueError('End year must be later than start year')

    if alignment_override is not None:
        alignment = _normalize_alignment_override(
            alignment_override,
            len(set(start.reference_site_id) & set(end.reference_site_id)),
        )
    else:
        alignment = (
            estimate_epoch_shift(
                start,
                end,
                min_alignment_parcels=min_alignment_parcels,
                area_ratio_tolerance=area_ratio_tolerance,
            )
            if align else _empty_alignment(
                len(set(start.reference_site_id) & set(end.reference_site_id))
            )
        )
    aligned_end = apply_epoch_shift(
        end, alignment['shift_dx_m'], alignment['shift_dy_m']
    )
    aligned_end.attrs.update(end.attrs)

    all_sites = pd.Index(start.reference_site_id).union(
        aligned_end.reference_site_id
    )
    start_indexed = start.set_index('reference_site_id')
    end_indexed = aligned_end.set_index('reference_site_id')
    start_geometry = gpd.GeoSeries(
        start_indexed.geometry.reindex(all_sites), index=all_sites, crs=start.crs
    )
    end_geometry = gpd.GeoSeries(
        end_indexed.geometry.reindex(all_sites), index=all_sites, crs=end.crs
    )
    has_start = _present_geometry(start_geometry)
    has_end = _present_geometry(end_geometry)
    both = has_start & has_end

    raw_added = gpd.GeoSeries([None] * len(all_sites), index=all_sites, crs=start.crs)
    raw_removed = raw_added.copy()
    stable = raw_added.copy()
    raw_added.loc[both] = end_geometry.loc[both].difference(
        start_geometry.loc[both], align=False
    ).to_numpy()
    raw_removed.loc[both] = start_geometry.loc[both].difference(
        end_geometry.loc[both], align=False
    ).to_numpy()
    stable.loc[both] = start_geometry.loc[both].intersection(
        end_geometry.loc[both], align=False
    ).to_numpy()
    raw_added.loc[~has_start & has_end] = end_geometry.loc[
        ~has_start & has_end
    ].to_numpy()
    raw_removed.loc[has_start & ~has_end] = start_geometry.loc[
        has_start & ~has_end
    ].to_numpy()

    clean_added = raw_added.copy()
    clean_removed = raw_removed.copy()
    clean_added.loc[both] = raw_added.loc[both].map(
        lambda geometry: remove_slivers(geometry, sliver_buffer_m)
    )
    clean_removed.loc[both] = raw_removed.loc[both].map(
        lambda geometry: remove_slivers(geometry, sliver_buffer_m)
    )
    filtered_added = _difference_series(raw_added, clean_added)
    filtered_removed = _difference_series(raw_removed, clean_removed)

    area_start = start_geometry.area.fillna(0.0)
    area_end = end_geometry.area.fillna(0.0)
    area_added_raw = raw_added.area.fillna(0.0)
    area_removed_raw = raw_removed.area.fillna(0.0)
    area_added = clean_added.area.fillna(0.0)
    area_removed = clean_removed.area.fillna(0.0)
    area_stable = stable.area.fillna(0.0)
    building_start = _reindex_building_count(start_indexed, all_sites)
    building_end = _reindex_building_count(end_indexed, all_sites)

    reference_ains = _reference_ains_for_sites(
        reference_parcels, all_sites, start.crs
    )

    metrics = pd.DataFrame(
        {
            'reference_site_id': all_sites,
            'reference_AIN': reference_ains.to_numpy(),
            'year_start': start_year,
            'year_end': end_year,
            'area_start_m2': area_start.to_numpy(),
            'area_end_m2': area_end.to_numpy(),
            'area_stable_m2': area_stable.to_numpy(),
            'area_added_raw_m2': area_added_raw.to_numpy(),
            'area_removed_raw_m2': area_removed_raw.to_numpy(),
            'area_added_m2': area_added.to_numpy(),
            'area_removed_m2': area_removed.to_numpy(),
            'area_filtered_added_m2': filtered_added.area.fillna(0.0).to_numpy(),
            'area_filtered_removed_m2': filtered_removed.area.fillna(0.0).to_numpy(),
            'building_count_start': building_start.to_numpy(),
            'building_count_end': building_end.to_numpy(),
        }
    )
    metrics['area_net_m2'] = metrics['area_added_m2'] - metrics['area_removed_m2']
    metrics['area_changed_m2'] = metrics['area_added_m2'] + metrics['area_removed_m2']
    metrics['building_count_delta'] = (
        metrics['building_count_end'] - metrics['building_count_start']
    )
    union_area = (
        metrics['area_start_m2'] + metrics['area_end_m2'] - metrics['area_stable_m2']
    )
    metrics['overlap_iou'] = metrics['area_stable_m2'].divide(
        union_area.where(union_area > 0)
    )
    maximum_area = metrics[['area_start_m2', 'area_end_m2']].max(axis=1)
    metrics['area_changed_fraction'] = metrics['area_changed_m2'].divide(
        maximum_area.where(maximum_area > 0)
    )
    metrics['change_type'] = _classify_changes(
        metrics, min_change_area_m2, relative_change_threshold
    )
    metrics['epoch'] = f'{start_year}→{end_year}'
    for key, value in alignment.items():
        metrics[key] = value

    parcel_geometry = _reference_geometry_for_metrics(
        reference_parcels, all_sites, start.crs
    )
    metrics = gpd.GeoDataFrame(metrics, geometry=parcel_geometry, crs=start.crs)
    numeric = metrics.select_dtypes(include='number').columns
    metrics[numeric] = metrics[numeric].round(3)
    diagnostics = {
        **alignment,
        'year_start': start_year,
        'year_end': end_year,
        'parcels_compared': int(len(metrics)),
        'changed_parcels': int(metrics.change_type.ne('no_change').sum()),
        'sliver_buffer_m': float(sliver_buffer_m),
        'min_change_area_m2': float(min_change_area_m2),
        'relative_change_threshold': float(relative_change_threshold),
        'min_alignment_parcels': int(min_alignment_parcels),
        'area_ratio_tolerance': float(area_ratio_tolerance),
    }
    metrics.attrs['change_diagnostics'] = diagnostics

    result = {'metrics': metrics, 'diagnostics': diagnostics}
    if store_geometries:
        result.update(
            {
                'raw_added': _geometry_output(metrics, raw_added, 'raw_added'),
                'raw_removed': _geometry_output(
                    metrics, raw_removed, 'raw_removed'
                ),
                'added': _geometry_output(metrics, clean_added, 'added'),
                'removed': _geometry_output(metrics, clean_removed, 'removed'),
                'filtered_added': _geometry_output(
                    metrics, filtered_added, 'filtered_added'
                ),
                'filtered_removed': _geometry_output(
                    metrics, filtered_removed, 'filtered_removed'
                ),
                'aligned_end': aligned_end,
            }
        )
    return result


def build_change_metrics(
    buildings_start: gpd.GeoDataFrame,
    buildings_end: gpd.GeoDataFrame,
    reference_parcels: gpd.GeoDataFrame,
    year_start: int,
    year_end: int,
    *,
    target_crs=DEFAULT_CHANGE_CRS,
    split_necks_m: Optional[float] = None,
    repair_boundary_artifacts=True,
    repair_options=None,
    **comparison_options,
) -> dict:
    """Prepare two building epochs and calculate parcel change metrics."""
    start = prepare_epoch_footprints(
        buildings_start,
        reference_parcels,
        year_start,
        target_crs=target_crs,
        split_necks_m=split_necks_m,
        repair_boundary_artifacts=repair_boundary_artifacts,
        repair_options=repair_options,
    )
    end = prepare_epoch_footprints(
        buildings_end,
        reference_parcels,
        year_end,
        target_crs=target_crs,
        split_necks_m=split_necks_m,
        repair_boundary_artifacts=repair_boundary_artifacts,
        repair_options=repair_options,
    )
    result = calculate_change_metrics(
        start,
        end,
        reference_parcels,
        **comparison_options,
    )
    result['footprints'] = {int(year_start): start, int(year_end): end}
    return result


def _prepare_reference_parcels(parcels: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if not isinstance(parcels, gpd.GeoDataFrame):
        raise TypeError('reference_parcels must be a GeoDataFrame')
    if 'AIN' not in parcels:
        raise ValueError('Reference parcels are missing AIN')
    parcels = add_parcel_site_ids(parcels)
    parcels = parcels[['site_id', 'AIN', parcels.geometry.name]].copy()
    if parcels.site_id.isna().any():
        raise ValueError('Reference parcel site_id values must be non-missing')
    if parcels.site_id.duplicated().any():
        parcels = parcels.dissolve(
            by='site_id', as_index=False, aggfunc='first'
        )
    return parcels.reset_index(drop=True)


def _validate_prepared_footprints(frame, label):
    required = {
        'reference_site_id', 'reference_AIN', 'year', 'footprint_area_m2',
        'building_count', 'footprint',
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f'{label} footprints are missing columns: {missing}')
    if frame.crs is None or frame.crs.is_geographic:
        raise ValueError(f'{label} footprints require a projected metric CRS')
    if frame.reference_site_id.duplicated().any():
        raise ValueError(
            f'{label} footprints must be unique by reference_site_id'
        )


def _single_year(frame, label):
    years = pd.to_numeric(frame['year'], errors='coerce').dropna().unique()
    if len(years) == 0 and 'year' in frame.attrs:
        return int(frame.attrs['year'])
    if len(years) != 1:
        raise ValueError(f'{label} footprints must contain exactly one year')
    return int(years[0])


def _empty_alignment(common_parcels, candidates=0):
    return {
        'shift_dx_m': 0.0,
        'shift_dy_m': 0.0,
        'common_sites': int(common_parcels),
        'common_parcels': int(common_parcels),
        'alignment_candidates': int(candidates),
        'alignment_used': 0,
        'alignment_residual_median_m': np.nan,
        'alignment_applied': False,
    }


def _normalize_alignment_override(alignment, common_sites):
    """Validate a precomputed shift for consistent multi-tile comparison."""
    try:
        dx = float(alignment['shift_dx_m'])
        dy = float(alignment['shift_dy_m'])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            'alignment_override requires numeric shift_dx_m and shift_dy_m'
        ) from error
    if not np.isfinite(dx) or not np.isfinite(dy):
        raise ValueError('alignment_override shifts must be finite')
    return {
        'shift_dx_m': dx,
        'shift_dy_m': dy,
        'common_sites': int(common_sites),
        'common_parcels': int(common_sites),
        'alignment_candidates': int(alignment.get('alignment_candidates', 0)),
        'alignment_used': int(alignment.get('alignment_used', 0)),
        'alignment_residual_median_m': float(
            alignment.get('alignment_residual_median_m', np.nan)
        ),
        'alignment_applied': True,
    }


def _difference_series(left, right):
    result = gpd.GeoSeries([None] * len(left), index=left.index, crs=left.crs)
    valid = _present_geometry(left)
    valid_right = _present_geometry(right)
    both = valid & valid_right
    if both.any():
        left_both = left.loc[both]
        right_both = right.loc[both]
        try:
            differences = left_both.difference(right_both, align=False).to_numpy()
        except GEOSException:
            # Overlay can fail on otherwise usable LARIAC polygons when
            # nearly coincident boundary segments are not fully noded.
            # ``make_valid`` can return polygon-plus-line collections, which
            # fixed-precision overlay rejects as mixed-dimension input. Keep
            # only polygonal components because these metrics measure area,
            # then snap at a micrometre grid for consistent topology.
            repaired_left = np.array(
                [_polygonal_make_valid(geometry) for geometry in left_both],
                dtype=object,
            )
            repaired_right = np.array(
                [_polygonal_make_valid(geometry) for geometry in right_both],
                dtype=object,
            )
            try:
                differences = shapely.difference(
                    repaired_left,
                    repaired_right,
                    grid_size=TOPOLOGY_GRID_SIZE_M,
                )
            except GEOSException:
                differences = np.array([
                    _precision_difference(left_geometry, right_geometry)
                    for left_geometry, right_geometry
                    in zip(repaired_left, repaired_right)
                ], dtype=object)
        result.loc[both] = differences
    result.loc[valid & ~valid_right] = left.loc[valid & ~valid_right].to_numpy()
    return result


def _polygonal_make_valid(geometry):
    """Repair a geometry and discard non-area components recursively."""
    repaired = shapely.make_valid(geometry)
    polygons = []
    pending = [repaired]
    while pending:
        current = pending.pop()
        if current is None or current.is_empty:
            continue
        if current.geom_type == 'Polygon':
            polygons.append(current)
        elif current.geom_type in ('MultiPolygon', 'GeometryCollection'):
            pending.extend(current.geoms)
    if not polygons:
        return shapely.GeometryCollection()
    if len(polygons) == 1:
        return polygons[0]
    return shapely.union_all(polygons)


def _precision_difference(left, right):
    """Retry one polygonal difference at increasingly coarse fine grids."""
    error = None
    for grid_size in (TOPOLOGY_GRID_SIZE_M, 1e-4, 1e-3):
        try:
            return shapely.difference(left, right, grid_size=grid_size)
        except GEOSException as caught:
            error = caught
    raise error


def _reindex_building_count(frame, index):
    values = pd.to_numeric(
        frame['building_count'].reindex(index), errors='coerce'
    ).to_numpy(dtype='float64', na_value=np.nan)
    return pd.Series(np.nan_to_num(values).astype('int64'), index=index)


def _classify_changes(metrics, minimum_area, relative_threshold):
    maximum_area = metrics[['area_start_m2', 'area_end_m2']].max(axis=1)
    threshold = np.maximum(minimum_area, relative_threshold * maximum_area)
    added = metrics['area_added_m2'] >= threshold
    removed = metrics['area_removed_m2'] >= threshold
    raw_changed = (
        metrics['area_added_raw_m2'] + metrics['area_removed_raw_m2']
    ) >= threshold
    no_start = metrics['area_start_m2'].eq(0)
    no_end = metrics['area_end_m2'].eq(0)
    count_delta = metrics['building_count_delta']

    result = pd.Series('no_change', index=metrics.index, dtype='string')
    result.loc[~added & ~removed & raw_changed] = 'possible_misalignment'
    result.loc[added & ~removed & count_delta.le(0)] = 'extended'
    result.loc[added & ~removed & count_delta.gt(0)] = 'added_structure'
    result.loc[~added & removed & count_delta.ge(0)] = 'reduced'
    result.loc[~added & removed & count_delta.lt(0)] = 'removed_structure'
    result.loc[added & removed] = 'replacement_or_reconfiguration'
    result.loc[no_start & added] = 'new_development'
    result.loc[no_end & removed] = 'demolished'
    return result


def _reference_geometry_for_metrics(reference_parcels, all_sites, target_crs):
    parcels = _prepare_reference_parcels(reference_parcels)
    if parcels.crs != target_crs:
        parcels = parcels.to_crs(target_crs)
    lookup = parcels.set_index('site_id').geometry
    return gpd.GeoSeries(
        lookup.reindex(all_sites).to_numpy(),
        index=pd.RangeIndex(len(all_sites)),
        crs=target_crs,
    )


def _reference_ains_for_sites(reference_parcels, all_sites, target_crs):
    parcels = _prepare_reference_parcels(reference_parcels)
    if parcels.crs != target_crs:
        parcels = parcels.to_crs(target_crs)
    return parcels.set_index('site_id')['AIN'].reindex(all_sites)


def _geometry_output(metrics, geometries, label):
    rows = metrics[
        [
            'reference_site_id', 'reference_AIN', 'year_start', 'year_end',
            'epoch', 'change_type',
        ]
    ].copy()
    rows['geometry'] = geometries.reindex(
        metrics.reference_site_id
    ).to_numpy()
    rows = gpd.GeoDataFrame(rows, geometry='geometry', crs=metrics.crs)
    rows = rows[_present_geometry(rows.geometry)].reset_index(drop=True)
    rows.attrs['geometry_role'] = label
    return rows


def _present_geometry(geometries):
    non_missing = pd.Series(
        ~pd.isna(geometries.array), index=geometries.index
    )
    return non_missing & ~geometries.is_empty
