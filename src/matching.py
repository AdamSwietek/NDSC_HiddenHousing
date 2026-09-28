"""Shared building-to-parcel matching and parcel aggregation."""

from __future__ import annotations

import warnings

import geopandas as gpd
import numpy as np
import pandas as pd

try:
    from .identifiers import normalize_ain_series
except ImportError:  # Support importing modules directly from the src directory.
    from identifiers import normalize_ain_series


MATCH_COLUMNS = [
    '_building_id',
    '_parcel_id',
    'match.method',
    'match.candidate_count',
    'match.overlap_ratio',
]


def aggregate_buildings_to_parcels(
    buildings: gpd.GeoDataFrame,
    parcels: gpd.GeoDataFrame,
    features: list[str],
    *,
    target_crs=None,
    return_matches=False,
):
    """Match buildings to parcels and aggregate numeric building features.

    Each building is assigned to at most one parcel. Buildings fully covered by
    one parcel use a fast representative-point assignment. Boundary-crossing,
    ambiguous, and initially unmatched buildings are resolved using their
    largest positive footprint overlap.

    The returned parcel GeoDataFrame includes parcels with zero buildings and
    retains parcel geometry. Global counts are stored in
    ``result.attrs['matching_diagnostics']``. With ``return_matches=True``, a
    third return value contains the one-row-per-building assignment table so
    callers can produce building and parcel outputs without matching twice.
    """
    buildings, parcels = align_matching_crs(buildings, parcels, target_crs)
    validate_matching_inputs(buildings, parcels, features)

    buildings = buildings.reset_index(drop=True).copy()
    parcels = parcels.reset_index(drop=True).copy()
    buildings['_building_id'] = np.arange(len(buildings), dtype=np.int64)
    parcels['_parcel_id'] = np.arange(len(parcels), dtype=np.int64)
    if 'AIN' in parcels.columns:
        parcels['AIN'] = normalize_ain_series(parcels['AIN'], name='AIN')

    valid_buildings = _present_geometry_mask(buildings)
    valid_parcels = _present_geometry_mask(parcels)
    matches = match_buildings_to_parcels(
        buildings.loc[valid_buildings].copy(),
        parcels.loc[valid_parcels].copy(),
    )

    out = gpd.GeoDataFrame(
        index=parcels['_parcel_id'],
        geometry=parcels.geometry.to_numpy(),
        crs=parcels.crs,
    )
    out.index.name = '_parcel_id'
    for key in ('site_id', 'AIN'):
        if key in parcels.columns:
            out[key] = parcels.set_index('_parcel_id')[key]

    matched = buildings.merge(matches, on='_building_id', how='inner')
    grouped = matched.groupby('_parcel_id', sort=False)
    if not matched.empty:
        aggregates = grouped[features].agg(['sum', 'max'])
        aggregates.columns = [
            f'{feature}.{operation}'
            for feature, operation in aggregates.columns
        ]
        out = out.join(aggregates.round(2))

    _add_empty_parcel_aggregates(out, features)
    building_counts = (
        grouped.size() if not matched.empty else pd.Series(dtype='int64')
    )
    out['shape.n_buildings'] = building_counts.reindex(
        out.index, fill_value=0
    ).astype('int64')
    _add_match_quality(out, matched, grouped)

    diagnostics = _matching_diagnostics(
        buildings, parcels, matches, valid_buildings, valid_parcels
    )
    out.attrs['matching_diagnostics'] = diagnostics

    aligned_parcels = parcels.drop(columns='_parcel_id')
    aligned_parcels.index = out.index
    aligned_parcels.index.name = out.index.name
    aligned_parcels.attrs['matching_diagnostics'] = diagnostics
    if return_matches:
        return out, aligned_parcels, matches
    return out, aligned_parcels


def unary_join_to_parcels(
    data: gpd.GeoDataFrame,
    parcel: gpd.GeoDataFrame,
    CRS,
    feats: list[str],
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Backward-compatible alias for :func:`aggregate_buildings_to_parcels`."""
    return aggregate_buildings_to_parcels(
        data, parcel, feats, target_crs=CRS
    )


def match_buildings_to_parcels(
    buildings: gpd.GeoDataFrame,
    parcels: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Return one deterministic parcel assignment per matched building.

    Inputs must contain unique integer ``_building_id`` and ``_parcel_id``
    columns and must already use the same CRS.
    """
    if buildings.empty or parcels.empty:
        return pd.DataFrame(columns=MATCH_COLUMNS)
    _validate_match_ids(buildings, parcels)
    if buildings.crs != parcels.crs:
        raise ValueError('Building and parcel CRS must match before assignment')

    parcel_keys = parcels[['_parcel_id', parcels.geometry.name]].copy()
    representative_points = gpd.GeoDataFrame(
        buildings[['_building_id']].copy(),
        geometry=buildings.geometry.representative_point(),
        crs=buildings.crs,
    )
    point_candidates = gpd.sjoin(
        representative_points,
        parcel_keys,
        how='left',
        predicate='within',
    )
    valid_candidates = point_candidates.dropna(subset=['_parcel_id']).copy()
    valid_candidates['_parcel_id'] = valid_candidates['_parcel_id'].astype('int64')
    candidate_counts = valid_candidates.groupby(
        '_building_id'
    )['_parcel_id'].nunique()
    unique_ids = candidate_counts[candidate_counts == 1].index
    unique_points = (
        valid_candidates[valid_candidates['_building_id'].isin(unique_ids)]
        .drop_duplicates('_building_id')
        [['_building_id', '_parcel_id']]
    )

    covered = _footprints_covered_by_parcels(unique_points, buildings, parcels)
    accepted = unique_points.loc[covered].copy()
    accepted['match.method'] = 'representative_point'
    accepted['match.candidate_count'] = 1
    accepted['match.overlap_ratio'] = 1.0

    accepted_ids = set(accepted['_building_id'])
    fallback = buildings.loc[~buildings['_building_id'].isin(accepted_ids)]
    resolved = _resolve_by_maximum_overlap(fallback, parcels)

    match_parts = [part for part in (accepted, resolved) if not part.empty]
    if not match_parts:
        return pd.DataFrame(columns=MATCH_COLUMNS)
    matches = pd.concat(match_parts, ignore_index=True)
    return (
        matches[MATCH_COLUMNS]
        .sort_values('_building_id')
        .reset_index(drop=True)
    )


def align_matching_crs(
    buildings: gpd.GeoDataFrame,
    parcels: gpd.GeoDataFrame,
    target_crs=None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Return copies transformed—not merely relabeled—to one CRS."""
    if not isinstance(buildings, gpd.GeoDataFrame):
        buildings = gpd.GeoDataFrame(buildings, geometry='geometry')
    if not isinstance(parcels, gpd.GeoDataFrame):
        parcels = gpd.GeoDataFrame(parcels, geometry='geometry')

    target_crs = target_crs or buildings.crs or parcels.crs
    if target_crs is None:
        raise ValueError('Building and parcel CRS are missing; provide target_crs')

    def align(frame: gpd.GeoDataFrame, label: str) -> gpd.GeoDataFrame:
        frame = frame.copy()
        if frame.crs is None:
            warnings.warn(
                f'{label} CRS was missing and has been set to {target_crs}; '
                'verify that this describes the source coordinates.',
                stacklevel=3,
            )
            return frame.set_crs(target_crs)
        if frame.crs != target_crs:
            return frame.to_crs(target_crs)
        return frame

    buildings = align(buildings, 'Building')
    parcels = align(parcels, 'Parcel')
    if buildings.crs.is_geographic:
        warnings.warn(
            'Matching in a geographic CRS is supported, but area-based metrics '
            'and overlap ratios should use a suitable projected CRS.',
            stacklevel=2,
        )
    return buildings, parcels


def validate_matching_inputs(
    buildings: gpd.GeoDataFrame,
    parcels: gpd.GeoDataFrame,
    features: list[str],
) -> None:
    """Validate feature columns and geometries before matching."""
    missing_features = sorted(set(features).difference(buildings.columns))
    if missing_features:
        raise ValueError(f'Missing building features: {missing_features}')
    non_numeric = [
        feature for feature in features
        if not pd.api.types.is_numeric_dtype(buildings[feature])
    ]
    if non_numeric:
        raise TypeError(f'Building features must be numeric: {non_numeric}')

    parcel_present = _present_geometry_mask(parcels)
    invalid_parcels = parcel_present & ~parcels.geometry.is_valid
    if invalid_parcels.any():
        raise ValueError(f'{int(invalid_parcels.sum())} parcel geometries are invalid')
    building_present = _present_geometry_mask(buildings)
    invalid_buildings = building_present & ~buildings.geometry.is_valid
    if invalid_buildings.any():
        raise ValueError(f'{int(invalid_buildings.sum())} building geometries are invalid')


def _footprints_covered_by_parcels(
    assignments: pd.DataFrame,
    buildings: gpd.GeoDataFrame,
    parcels: gpd.GeoDataFrame,
) -> pd.Series:
    if assignments.empty:
        return pd.Series(dtype=bool, index=assignments.index)
    building_geometries = gpd.GeoSeries(
        buildings.set_index('_building_id').geometry
        .loc[assignments['_building_id']].to_numpy(),
        index=assignments.index,
        crs=buildings.crs,
    )
    parcel_geometries = gpd.GeoSeries(
        parcels.set_index('_parcel_id').geometry
        .loc[assignments['_parcel_id']].to_numpy(),
        index=assignments.index,
        crs=parcels.crs,
    )
    return building_geometries.covered_by(parcel_geometries, align=False)


def _present_geometry_mask(frame: gpd.GeoDataFrame) -> pd.Series:
    """Return true for non-missing, non-empty geometries without API warnings."""
    non_missing = pd.Series(
        ~pd.isna(frame.geometry.array),
        index=frame.index,
    )
    return non_missing & ~frame.geometry.is_empty


def _resolve_by_maximum_overlap(
    buildings: gpd.GeoDataFrame,
    parcels: gpd.GeoDataFrame,
) -> pd.DataFrame:
    if buildings.empty:
        return pd.DataFrame(columns=MATCH_COLUMNS)

    candidates = gpd.sjoin(
        buildings[['_building_id', buildings.geometry.name]],
        parcels[['_parcel_id', parcels.geometry.name]],
        how='left',
        predicate='intersects',
    ).dropna(subset=['_parcel_id'])
    if candidates.empty:
        return pd.DataFrame(columns=MATCH_COLUMNS)

    candidates['_parcel_id'] = candidates['_parcel_id'].astype('int64')
    building_geometries = gpd.GeoSeries(
        candidates.geometry.to_numpy(),
        index=candidates.index,
        crs=buildings.crs,
    )
    parcel_lookup = parcels.set_index('_parcel_id').geometry
    parcel_geometries = gpd.GeoSeries(
        parcel_lookup.loc[candidates['_parcel_id']].to_numpy(),
        index=candidates.index,
        crs=parcels.crs,
    )
    candidates['match.overlap_area'] = building_geometries.intersection(
        parcel_geometries, align=False
    ).area
    candidates['match.building_area'] = building_geometries.area
    candidates = candidates[candidates['match.overlap_area'] > 0].copy()
    if candidates.empty:
        return pd.DataFrame(columns=MATCH_COLUMNS)

    candidates['match.candidate_count'] = candidates.groupby(
        '_building_id'
    )['_parcel_id'].transform('nunique')
    candidates['match.overlap_ratio'] = (
        candidates['match.overlap_area'] / candidates['match.building_area']
    )
    candidates['match.method'] = 'maximum_overlap'
    candidates = candidates.sort_values(
        ['_building_id', 'match.overlap_area', '_parcel_id'],
        ascending=[True, False, True],
    )
    return candidates.drop_duplicates('_building_id')[MATCH_COLUMNS]


def _add_empty_parcel_aggregates(
    out: gpd.GeoDataFrame,
    features: list[str],
) -> None:
    for column in (f'{feature}.sum' for feature in features):
        if column not in out:
            out[column] = 0.0
        else:
            out[column] = out[column].fillna(0.0)
    for column in (f'{feature}.max' for feature in features):
        if column not in out:
            out[column] = np.nan


def _add_match_quality(out, matched, grouped) -> None:
    if matched.empty:
        out['match.overlap_ratio.min'] = np.nan
        out['match.n_overlap_resolved'] = 0
        return
    out['match.overlap_ratio.min'] = grouped['match.overlap_ratio'].min().reindex(
        out.index
    )
    overlap_resolved = (
        matched['match.method'].eq('maximum_overlap')
        .groupby(matched['_parcel_id'])
        .sum()
    )
    out['match.n_overlap_resolved'] = overlap_resolved.reindex(
        out.index, fill_value=0
    ).astype('int64')


def _matching_diagnostics(buildings, parcels, matches, valid_buildings, valid_parcels):
    matched_count = matches['_building_id'].nunique()
    return {
        'input_buildings': int(len(buildings)),
        'valid_buildings': int(valid_buildings.sum()),
        'matched_buildings': int(matched_count),
        'unmatched_buildings': int(valid_buildings.sum() - matched_count),
        'invalid_or_empty_buildings': int((~valid_buildings).sum()),
        'representative_point_matches': int(
            matches['match.method'].eq('representative_point').sum()
        ),
        'maximum_overlap_matches': int(
            matches['match.method'].eq('maximum_overlap').sum()
        ),
        'input_parcels': int(len(parcels)),
        'invalid_or_empty_parcels': int((~valid_parcels).sum()),
    }


def _validate_match_ids(buildings, parcels) -> None:
    for frame, column in (
        (buildings, '_building_id'),
        (parcels, '_parcel_id'),
    ):
        if column not in frame:
            raise ValueError(f'Missing required match column: {column}')
        if frame[column].isna().any() or frame[column].duplicated().any():
            raise ValueError(f'{column} must contain unique non-missing values')
