"""Conservative repairs for source-level LARIAC footprint artifacts."""

from __future__ import annotations

import hashlib

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely


DEFAULT_MAX_FRAGMENT_AREA_FT2 = 500.0
DEFAULT_MIN_SHARED_EDGE_FT = 2.0
DEFAULT_BOUNDARY_TOLERANCE_FT = 0.25
DEFAULT_MIN_BOUNDARY_COVERAGE = 0.80
DEFAULT_MAX_HEIGHT_DIFFERENCE_FT = 1.0
DEFAULT_MAX_ELEVATION_DIFFERENCE_FT = 0.5
DEFAULT_SLIVER_BUFFER_FT = 2.0
DEFAULT_MAX_OPENED_RETAINED_FRACTION = 0.10


def repair_boundary_fragments(
    buildings: gpd.GeoDataFrame,
    parcels: gpd.GeoDataFrame,
    *,
    max_fragment_area_ft2: float = DEFAULT_MAX_FRAGMENT_AREA_FT2,
    min_shared_edge_ft: float = DEFAULT_MIN_SHARED_EDGE_FT,
    boundary_tolerance_ft: float = DEFAULT_BOUNDARY_TOLERANCE_FT,
    min_boundary_coverage: float = DEFAULT_MIN_BOUNDARY_COVERAGE,
    max_height_difference_ft: float = DEFAULT_MAX_HEIGHT_DIFFERENCE_FT,
    max_elevation_difference_ft: float = DEFAULT_MAX_ELEVATION_DIFFERENCE_FT,
    sliver_buffer_ft: float = DEFAULT_SLIVER_BUFFER_FT,
    max_opened_retained_fraction: float = DEFAULT_MAX_OPENED_RETAINED_FRACTION,
) -> gpd.GeoDataFrame:
    """Merge small parcel-boundary fragments into their source footprint.

    LARIAC occasionally represents one physical structure as a large polygon
    plus tiny, separately identified polygons on the other side of a parcel
    boundary. A fragment is absorbed only when it shares a line segment with a
    footprint assigned to the parcel on the other side, that segment follows
    their parcel boundary, height and elevation agree, and a morphological
    opening removes nearly all of the candidate. Optional source and capture-
    date fields must also agree when both records contain them.

    Thresholds are expressed in feet and square feet, then converted to the
    horizontal units of the projected input CRS. The dominant record supplies
    the reconstructed structure's attributes. Repair lineage and diagnostics
    are retained on the returned frame.
    """
    _validate_inputs(buildings, parcels)
    _validate_thresholds(
        max_fragment_area_ft2,
        min_shared_edge_ft,
        boundary_tolerance_ft,
        min_boundary_coverage,
        max_height_difference_ft,
        max_elevation_difference_ft,
        sliver_buffer_ft,
        max_opened_retained_fraction,
    )
    if min_boundary_coverage > 1 or max_opened_retained_fraction > 1:
        raise ValueError('Coverage and retained-fraction thresholds cannot exceed 1')

    original_attrs = dict(buildings.attrs)
    frame = buildings.copy().reset_index(drop=True)
    frame['structure_id'] = _structure_ids(frame)
    frame['repair.fragment_count'] = 0
    frame['repair.absorbed_area_ft2'] = 0.0
    frame['repair.source_building_ids'] = [
        (str(value),) for value in frame['structure_id']
    ]
    frame['repair.method'] = 'none'

    diagnostics = {
        'input_buildings': int(len(frame)),
        'output_structures': int(len(frame)),
        'candidate_fragments': 0,
        'merged_fragments': 0,
        'reconstructed_structures': 0,
        'absorbed_area_ft2': 0.0,
        'skipped_reason': None,
    }
    required_attributes = {'height', 'elevation'}
    if frame.empty:
        return _finish(frame, original_attrs, diagnostics, [])
    if not required_attributes.issubset(frame.columns):
        diagnostics['skipped_reason'] = 'missing height or elevation'
        return _finish(frame, original_attrs, diagnostics, [])

    metres_per_unit = _metres_per_horizontal_unit(frame.crs)
    feet_to_units = 0.3048 / metres_per_unit
    square_feet_to_units = feet_to_units**2
    max_fragment_area = max_fragment_area_ft2 * square_feet_to_units
    min_shared_edge = min_shared_edge_ft * feet_to_units
    boundary_tolerance = boundary_tolerance_ft * feet_to_units
    sliver_buffer = sliver_buffer_ft * feet_to_units

    areas = frame.geometry.area
    valid = (
        frame.geometry.notna()
        & ~frame.geometry.is_empty
        & frame.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])
    )
    fragment_mask = valid & areas.gt(0) & areas.le(max_fragment_area)
    diagnostics['candidate_fragments'] = int(fragment_mask.sum())
    if not fragment_mask.any():
        return _finish(frame, original_attrs, diagnostics, [])

    fragments = frame.loc[fragment_mask, [frame.geometry.name]].copy()
    fragments['_fragment_index'] = fragments.index
    candidates = frame.loc[valid, [frame.geometry.name]].copy()
    candidates['_dominant_index'] = candidates.index
    pairs = gpd.sjoin(
        fragments,
        candidates,
        how='inner',
        predicate='intersects',
        lsuffix='fragment',
        rsuffix='dominant',
    )
    if pairs.empty:
        return _finish(frame, original_attrs, diagnostics, [])

    parcel_boundaries = parcels.geometry.boundary.reset_index(drop=True)
    boundary_index = parcel_boundaries.sindex
    representative_parcels = _representative_parcel_indices(frame, parcels)
    accepted = []
    for _, pair in pairs.iterrows():
        fragment_index = int(pair['_fragment_index'])
        dominant_index = int(pair['_dominant_index'])
        if fragment_index == dominant_index:
            continue
        fragment_area = float(areas.iloc[fragment_index])
        dominant_area = float(areas.iloc[dominant_index])
        if dominant_area <= fragment_area:
            continue
        fragment_parcel = representative_parcels.iloc[fragment_index]
        dominant_parcel = representative_parcels.iloc[dominant_index]
        if (
            pd.isna(fragment_parcel)
            or pd.isna(dominant_parcel)
            or int(fragment_parcel) == int(dominant_parcel)
        ):
            continue
        if not _attributes_agree(
            frame.iloc[fragment_index],
            frame.iloc[dominant_index],
            max_height_difference_ft,
            max_elevation_difference_ft,
        ):
            continue

        fragment_geometry = frame.geometry.iloc[fragment_index]
        dominant_geometry = frame.geometry.iloc[dominant_index]
        shared_edge = fragment_geometry.boundary.intersection(
            dominant_geometry.boundary
        )
        shared_length = float(shared_edge.length)
        if shared_length < min_shared_edge:
            continue
        boundary_coverage = _parcel_boundary_coverage(
            shared_edge,
            parcel_boundaries,
            boundary_index,
            boundary_tolerance,
        )
        if boundary_coverage < min_boundary_coverage:
            continue
        opened_retained_fraction = _opened_retained_fraction(
            fragment_geometry, sliver_buffer
        )
        if opened_retained_fraction > max_opened_retained_fraction:
            continue
        accepted.append(
            {
                'fragment_index': fragment_index,
                'dominant_index': dominant_index,
                'fragment_area': fragment_area,
                'shared_edge': shared_length,
                'boundary_coverage': boundary_coverage,
                'fragment_parcel_index': int(fragment_parcel),
                'dominant_parcel_index': int(dominant_parcel),
                'fragment_AIN': _parcel_label(parcels, fragment_parcel),
                'dominant_AIN': _parcel_label(parcels, dominant_parcel),
                'opened_retained_fraction': opened_retained_fraction,
            }
        )

    if not accepted:
        return _finish(frame, original_attrs, diagnostics, [])

    accepted_frame = pd.DataFrame(accepted).sort_values(
        ['fragment_index', 'shared_edge', 'fragment_area', 'dominant_index'],
        ascending=[True, False, True, True],
    )
    accepted_frame = accepted_frame.drop_duplicates('fragment_index')
    repairs = []
    absorbed_indices = set()
    for dominant_index, group in accepted_frame.groupby('dominant_index'):
        fragment_indices = group['fragment_index'].astype(int).tolist()
        absorbed_indices.update(fragment_indices)
        merged_geometry = shapely.union_all(
            [frame.geometry.iloc[int(dominant_index)]]
            + [frame.geometry.iloc[index] for index in fragment_indices]
        )
        frame.at[int(dominant_index), frame.geometry.name] = merged_geometry
        absorbed_area_ft2 = float(
            group['fragment_area'].sum() / square_feet_to_units
        )
        source_ids = [str(frame.at[int(dominant_index), 'structure_id'])]
        source_ids.extend(
            str(frame.at[index, 'structure_id']) for index in fragment_indices
        )
        frame.at[int(dominant_index), 'repair.fragment_count'] = len(fragment_indices)
        frame.at[int(dominant_index), 'repair.absorbed_area_ft2'] = absorbed_area_ft2
        frame.at[int(dominant_index), 'repair.source_building_ids'] = tuple(source_ids)
        frame.at[int(dominant_index), 'repair.method'] = 'boundary_fragment_merge'
        for row in group.itertuples(index=False):
            repairs.append(
                {
                    'fragment_id': str(
                        frame.at[int(row.fragment_index), 'structure_id']
                    ),
                    'dominant_id': str(
                        frame.at[int(dominant_index), 'structure_id']
                    ),
                    'fragment_area_ft2': float(
                        row.fragment_area / square_feet_to_units
                    ),
                    'shared_edge_ft': float(row.shared_edge / feet_to_units),
                    'parcel_boundary_coverage': float(row.boundary_coverage),
                    'fragment_parcel_index': int(row.fragment_parcel_index),
                    'dominant_parcel_index': int(row.dominant_parcel_index),
                    'fragment_AIN': row.fragment_AIN,
                    'dominant_AIN': row.dominant_AIN,
                    'sliver_buffer_ft': float(sliver_buffer_ft),
                    'opened_retained_fraction': float(
                        row.opened_retained_fraction
                    ),
                }
            )

    frame = frame.drop(index=sorted(absorbed_indices)).reset_index(drop=True)
    diagnostics.update(
        {
            'output_structures': int(len(frame)),
            'merged_fragments': int(len(absorbed_indices)),
            'reconstructed_structures': int(
                accepted_frame['dominant_index'].nunique()
            ),
            'absorbed_area_ft2': float(
                sum(record['fragment_area_ft2'] for record in repairs)
            ),
        }
    )
    return _finish(frame, original_attrs, diagnostics, repairs)


def _validate_inputs(buildings, parcels):
    if not isinstance(buildings, gpd.GeoDataFrame):
        raise TypeError('buildings must be a GeoDataFrame')
    if not isinstance(parcels, gpd.GeoDataFrame):
        raise TypeError('parcels must be a GeoDataFrame')
    if buildings.crs is None or buildings.crs.is_geographic:
        raise ValueError('Boundary repair requires a projected building CRS')
    if parcels.crs != buildings.crs:
        raise ValueError('Building and parcel CRS values must match before repair')


def _validate_thresholds(*values):
    if any(not np.isfinite(value) or value < 0 for value in values):
        raise ValueError('Boundary-repair thresholds must be finite and non-negative')


def _metres_per_horizontal_unit(crs) -> float:
    axis_info = crs.axis_info
    if not axis_info or not axis_info[0].unit_conversion_factor:
        raise ValueError('Unable to determine projected CRS units')
    return float(axis_info[0].unit_conversion_factor)


def _structure_ids(frame):
    if 'building_id' in frame:
        values = frame['building_id'].astype('string')
        missing = values.isna()
        if missing.any():
            values.loc[missing] = [
                f'geometry-{_geometry_fingerprint(frame.geometry.loc[index])}'
                for index in frame.index[missing]
            ]
    else:
        values = pd.Series(
            [
                f'geometry-{_geometry_fingerprint(geometry)}'
                for geometry in frame.geometry
            ],
            index=frame.index,
            dtype='string',
        )

    # BLD_ID is not globally unique in the LARIAC source. Preserve the source
    # value for normal records, but distinguish genuinely different geometries
    # that reuse it. A geometry-derived suffix remains stable when overlapping
    # processing tiles load the same feature independently.
    reused = values.duplicated(keep=False)
    if reused.any():
        values.loc[reused] = [
            f'{value}~{_geometry_fingerprint(geometry)}'
            for value, geometry in zip(
                values.loc[reused], frame.loc[reused, frame.geometry.name]
            )
        ]

    # Exact duplicate source rows can still produce the same fingerprint. Keep
    # IDs unique for validation; normal GDB read order makes the ordinal stable
    # across overlapping tile reads.
    collisions = values.duplicated(keep=False)
    if collisions.any():
        occurrences = values.loc[collisions].groupby(values.loc[collisions]).cumcount()
        values.loc[collisions] = values.loc[collisions] + '~' + occurrences.astype(str)
    return values


def _geometry_fingerprint(geometry):
    if geometry is None or geometry.is_empty:
        return 'missing'
    canonical = shapely.normalize(geometry)
    wkb = shapely.to_wkb(canonical, byte_order=1, output_dimension=2)
    return hashlib.blake2b(wkb, digest_size=8).hexdigest()


def _attributes_agree(
    fragment,
    dominant,
    max_height_difference_ft,
    max_elevation_difference_ft,
):
    for column, tolerance in (
        ('height', max_height_difference_ft),
        ('elevation', max_elevation_difference_ft),
    ):
        left = pd.to_numeric(pd.Series([fragment[column]]), errors='coerce').iloc[0]
        right = pd.to_numeric(pd.Series([dominant[column]]), errors='coerce').iloc[0]
        if pd.isna(left) or pd.isna(right) or abs(float(left) - float(right)) > tolerance:
            return False
    for column in ('source', 'capture_date'):
        if column not in fragment.index:
            continue
        left = fragment[column]
        right = dominant[column]
        if pd.notna(left) and pd.notna(right) and str(left) != str(right):
            return False
    return True


def _representative_parcel_indices(frame, parcels):
    """Return one containing parcel index for each footprint representative point."""
    points = gpd.GeoDataFrame(
        {'_building_index': frame.index},
        geometry=frame.geometry.representative_point(),
        crs=frame.crs,
    )
    parcel_keys = gpd.GeoDataFrame(
        {'_representative_parcel_index': np.arange(len(parcels), dtype=np.int64)},
        geometry=parcels.geometry.to_numpy(),
        crs=parcels.crs,
    )
    joined = gpd.sjoin(points, parcel_keys, how='left', predicate='within')
    valid = joined.dropna(subset=['_representative_parcel_index']).copy()
    unique_counts = valid.groupby('_building_index')[
        '_representative_parcel_index'
    ].nunique()
    unique_buildings = unique_counts[unique_counts.eq(1)].index
    assignments = (
        valid[valid['_building_index'].isin(unique_buildings)]
        .drop_duplicates('_building_index')
        .set_index('_building_index')['_representative_parcel_index']
    )
    return assignments.reindex(frame.index).astype('Int64')


def _parcel_boundary_coverage(
    shared_edge,
    parcel_boundaries,
    boundary_index,
    tolerance,
):
    positions = boundary_index.query(shared_edge.buffer(tolerance), predicate='intersects')
    if len(positions) == 0:
        return 0.0
    nearby = parcel_boundaries.iloc[np.asarray(positions, dtype=int)]
    boundary_zone = shapely.union_all(nearby.to_numpy()).buffer(tolerance)
    return float(shared_edge.intersection(boundary_zone).length / shared_edge.length)


def _parcel_label(parcels, parcel_index):
    if 'AIN' not in parcels:
        return None
    value = parcels.iloc[int(parcel_index)]['AIN']
    return None if pd.isna(value) else str(value)


def _opened_retained_fraction(geometry, buffer_distance):
    if buffer_distance == 0:
        return 1.0
    opened = geometry.buffer(-buffer_distance, join_style=2)
    if opened.is_empty:
        return 0.0
    opened = opened.buffer(buffer_distance, join_style=2)
    return float(opened.intersection(geometry).area / geometry.area)


def _finish(frame, original_attrs, diagnostics, repairs):
    result = gpd.GeoDataFrame(frame, geometry=frame.geometry.name, crs=frame.crs)
    result.attrs.update(original_attrs)
    result.attrs['boundary_repair_diagnostics'] = diagnostics
    result.attrs['boundary_repair_records'] = repairs
    return result
