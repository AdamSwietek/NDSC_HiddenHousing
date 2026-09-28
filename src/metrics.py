import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

try:
    from .identifiers import normalize_ain_series
    from .sites import add_parcel_site_ids
    from .lariac_cleaning import repair_boundary_fragments
    from .matching import (
        aggregate_buildings_to_parcels,
        align_matching_crs,
        match_buildings_to_parcels,
        unary_join_to_parcels,
    )
except ImportError:  # Support importing modules directly from the src directory.
    from identifiers import normalize_ain_series
    from sites import add_parcel_site_ids
    from lariac_cleaning import repair_boundary_fragments
    from matching import (
        aggregate_buildings_to_parcels,
        align_matching_crs,
        match_buildings_to_parcels,
        unary_join_to_parcels,
    )

FLOOR_HEIGHT_FT = 10 #ft

BUILDING_FEATURE_COLUMNS = [
    'height',
    'z.min',
    'z.max',
    'shape.area',
    'shape.length',
    'shape.circularity',
    'shape.fractality',
    'shape.rectangularity',
    'shape.convexity',
    'shape.squareness',
    'shape.exchange',
    'shape.perimeter',
    'shape.volume',
    'shape.nfloors',
    'shape.floorarea',
]


def get_building_lvl_feats(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Calculate shape and height features for every building.

    Calculations are performed on whole GeoSeries/Series instead of iterating
    over rows. Empty or missing geometries are omitted, matching the previous
    behavior. Both Polygon and MultiPolygon footprints are supported.

    The input must already use a projected CRS whose horizontal units match
    ``height`` and ``FLOOR_HEIGHT_FT`` (normally feet).
    """
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError("gdf must be a GeoDataFrame")

    required = {'height', 'elevation'}
    missing = required.difference(gdf.columns)
    if missing:
        raise ValueError(f"Missing required building columns: {sorted(missing)}")

    valid_geometry = pd.Series(
        ~pd.isna(gdf.geometry.array), index=gdf.index
    ) & ~gdf.geometry.is_empty
    out = gdf.loc[valid_geometry].copy()

    non_polygon = ~out.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])
    if non_polygon.any():
        bad_indices = out.index[non_polygon].tolist()[:5]
        raise TypeError(
            "Building geometries must be Polygon or MultiPolygon; "
            f"invalid rows include {bad_indices}"
        )

    if out.empty:
        for column in BUILDING_FEATURE_COLUMNS:
            if column not in out.columns:
                out[column] = pd.Series(index=out.index, dtype='float64')
        return out

    shape = out.geometry
    height = pd.to_numeric(out['height'], errors='coerce')
    elevation = pd.to_numeric(out['elevation'], errors='coerce')

    area = shape.area
    length = shape.length
    min_rectangles = shape.minimum_rotated_rectangle
    # GeoPandas has exposed this operation as both a property and a method
    # across releases. Shapely 2.1 can emit harmless floating-point warnings
    # for axis-aligned rectangles even though it returns the correct geometry.
    if callable(min_rectangles):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message='.*encountered in oriented_envelope',
                category=RuntimeWarning,
            )
            min_rectangles = min_rectangles()
    min_rectangle_area = min_rectangles.area
    convex_hull_area = shape.convex_hull.area

    with np.errstate(divide='ignore', invalid='ignore'):
        log_length = np.log(length)
        floors = np.floor(height / FLOOR_HEIGHT_FT).astype('Int64')
        features = pd.DataFrame(
            {
                'z.min': elevation,
                'z.max': elevation + height,
                'shape.area': area,
                'shape.length': length,
                'shape.circularity': _safe_divide(
                    4.0 * np.pi * area, length**2
                ),
                'shape.fractality': 1.0 - _safe_divide(
                    np.log(area), 2.0 * log_length
                ),
                'shape.rectangularity': _safe_divide(
                    area, min_rectangle_area
                ),
                'shape.convexity': _safe_divide(area, convex_hull_area),
                'shape.squareness': _safe_divide(
                    4.0 * np.sqrt(area), length
                ),
                'shape.exchange': _exchange_2d_series(shape, area),
                # This is a normalized compactness measure retained under its
                # existing name for backward compatibility.
                'shape.perimeter': _safe_divide(
                    2.0 * np.sqrt(np.pi * area), length
                ),
                'shape.volume': area * height,
                'shape.nfloors': floors,
                'shape.floorarea': area * floors,
            },
            index=out.index,
        )

    # Make the operation repeatable if a previously enriched frame is passed.
    existing_features = features.columns.intersection(out.columns)
    if len(existing_features):
        out = out.drop(columns=existing_features)

    # ``height`` already exists on the input, so concatenating only the newly
    # derived columns avoids creating duplicate column names.
    out = pd.concat([out, features], axis=1)
    return gpd.GeoDataFrame(out, geometry=gdf.geometry.name, crs=gdf.crs)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide aligned Series and return NaN where the denominator is zero."""
    result = numerator / denominator
    return result.where(denominator != 0)


def _exchange_2d_series(
    shapes: gpd.GeoSeries, areas: pd.Series
) -> pd.Series:
    """Vectorized equal-area-circle overlap for a GeoSeries of footprints."""
    radii = np.sqrt(areas / np.pi)
    circles = shapes.centroid.buffer(radii)
    try:
        intersection_area = shapes.intersection(circles).area
        return _safe_divide(intersection_area, areas)
    except Exception:
        # Preserve the scalar helper's per-geometry error handling if one
        # invalid footprint prevents the vectorized topology operation.
        return shapes.map(exchange_2d)


PARCEL_AGGREGATE_FEATURES = [
    'shape.area',
    'shape.volume',
    'shape.floorarea',
]

ASSESSOR_FIELD_NAMES = {
    'SQFTmain': 'assessor.sqft',
    'Units': 'assessor.units',
    'Bedrooms': 'assessor.bedrooms',
    'Bathrooms': 'assessor.bathrooms',
    'YearBuilt': 'assessor.year_built',
    'EffectiveYearBuilt': 'assessor.effective_year_built',
    'totBuildingDataLines': 'assessor.building_records',
}


def build_snapshot_metrics(
    buildings: gpd.GeoDataFrame,
    parcels: gpd.GeoDataFrame,
    assessor: pd.DataFrame,
    year: int,
    *,
    target_crs=None,
    repair_boundary_artifacts=True,
    repair_options=None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Build building and parcel snapshot features for one survey year.

    Source-level parcel-boundary fragments are repaired before features and
    assignment unless ``repair_boundary_artifacts`` is false. Buildings are
    then matched exactly once. The first result retains every valid structure,
    including unmatched structures with missing parcel fields. The second
    result has one row per canonical AIN and combines spatial totals with the
    same-year assessor roll.
    """
    year = int(year)
    _validate_snapshot_year(buildings, 'lariac_year', year, 'LARIAC')
    _validate_snapshot_year(assessor, 'assessor_year', year, 'assessor')

    parcels = _prepare_snapshot_parcels(parcels)
    buildings, parcels = align_matching_crs(buildings, parcels, target_crs)
    if buildings.crs is None or buildings.crs.is_geographic:
        raise ValueError('Snapshot metrics require a projected target_crs')
    buildings = _clean_snapshot_building_geometries(buildings)
    geometry_cleaning = buildings.attrs.get('geometry_cleaning_diagnostics', {})
    if repair_boundary_artifacts:
        buildings = repair_boundary_fragments(
            buildings, parcels, **(repair_options or {})
        )
    repair_diagnostics = buildings.attrs.get('boundary_repair_diagnostics')
    repair_records = buildings.attrs.get('boundary_repair_records', [])
    building_metrics = get_building_lvl_feats(buildings).reset_index(drop=True)

    parcel_aggregates, aligned_parcels, matches = (
        aggregate_buildings_to_parcels(
            building_metrics,
            parcels,
            PARCEL_AGGREGATE_FEATURES,
            target_crs=target_crs,
            return_matches=True,
        )
    )
    diagnostics = parcel_aggregates.attrs['matching_diagnostics']

    building_metrics['_building_id'] = np.arange(
        len(building_metrics), dtype=np.int64
    )
    building_metrics = building_metrics.merge(
        matches, on='_building_id', how='left', validate='one_to_one'
    )
    parcel_lookup = (
        parcel_aggregates[['site_id', 'AIN']]
        .reset_index()
        [['_parcel_id', 'site_id', 'AIN']]
    )
    building_metrics = building_metrics.merge(
        parcel_lookup, on='_parcel_id', how='left', validate='many_to_one'
    )
    if repair_records and 'structure_id' in building_metrics:
        structure_sites = building_metrics.set_index('structure_id')['site_id']
        repair_records = [
            {
                **record,
                'dominant_site_id': structure_sites.get(record['dominant_id']),
            }
            for record in repair_records
        ]
    building_metrics['snapshot_year'] = year
    building_metrics = gpd.GeoDataFrame(
        building_metrics, geometry=buildings.geometry.name,
        crs=parcel_aggregates.crs,
    )
    building_metrics.attrs['matching_diagnostics'] = diagnostics
    building_metrics.attrs['geometry_cleaning_diagnostics'] = geometry_cleaning
    if repair_diagnostics is not None:
        building_metrics.attrs['boundary_repair_diagnostics'] = repair_diagnostics
        building_metrics.attrs['boundary_repair_records'] = repair_records

    parcel_metrics = _build_parcel_snapshot(
        parcel_aggregates, aligned_parcels, assessor, year
    )
    parcel_metrics.attrs['matching_diagnostics'] = diagnostics
    parcel_metrics.attrs['geometry_cleaning_diagnostics'] = geometry_cleaning
    if repair_diagnostics is not None:
        parcel_metrics.attrs['boundary_repair_diagnostics'] = repair_diagnostics
        parcel_metrics.attrs['boundary_repair_records'] = repair_records
    return building_metrics, parcel_metrics


def _clean_snapshot_building_geometries(buildings):
    """Repair invalid source polygons before feature calculation and matching."""
    original_attrs = dict(buildings.attrs)
    frame = buildings.copy()
    present = pd.Series(
        ~pd.isna(frame.geometry.array), index=frame.index
    ) & ~frame.geometry.is_empty
    invalid = present & ~frame.geometry.is_valid
    if invalid.any():
        frame.loc[invalid, frame.geometry.name] = frame.loc[
            invalid, frame.geometry.name
        ].map(_polygonal_make_valid)
    retained = pd.Series(
        ~pd.isna(frame.geometry.array), index=frame.index
    ) & ~frame.geometry.is_empty & frame.geometry.geom_type.isin(
        ['Polygon', 'MultiPolygon']
    )
    result = frame.loc[retained].reset_index(drop=True)
    result.attrs.update(original_attrs)
    result.attrs['geometry_cleaning_diagnostics'] = {
        'input_buildings': int(len(frame)),
        'invalid_buildings_repaired': int(invalid.sum()),
        'non_polygon_or_empty_buildings_dropped': int((~retained).sum()),
        'output_buildings': int(len(result)),
    }
    return result


def _polygonal_make_valid(geometry):
    if geometry is None or geometry.is_empty:
        return geometry
    repaired = shapely.make_valid(geometry)
    if repaired.geom_type in {'Polygon', 'MultiPolygon'}:
        return repaired
    polygon_parts = [
        part for part in shapely.get_parts(repaired)
        if part.geom_type in {'Polygon', 'MultiPolygon'} and not part.is_empty
    ]
    return shapely.union_all(polygon_parts) if polygon_parts else None


def _prepare_snapshot_parcels(parcels: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if not isinstance(parcels, gpd.GeoDataFrame):
        raise TypeError('parcels must be a GeoDataFrame')
    if 'AIN' not in parcels:
        raise ValueError('Parcels are missing AIN')
    parcels = add_parcel_site_ids(parcels)
    if parcels['site_id'].isna().any():
        raise ValueError('Parcel site_id values must be non-missing')
    if parcels['site_id'].duplicated().any():
        parcels = parcels.dissolve(by='site_id', as_index=False, aggfunc='first')
    return parcels.reset_index(drop=True)


def _build_parcel_snapshot(
    aggregates: gpd.GeoDataFrame,
    aligned_parcels: gpd.GeoDataFrame,
    assessor: pd.DataFrame,
    year: int,
) -> gpd.GeoDataFrame:
    out = aggregates.copy()
    parcel_attributes = aligned_parcels.drop(
        columns=aligned_parcels.geometry.name, errors='ignore'
    )
    for column in parcel_attributes.columns:
        if column not in out:
            out[column] = parcel_attributes[column].to_numpy()
    out['snapshot_year'] = year
    out['parcel.area'] = out.geometry.area
    out['building.count'] = out['shape.n_buildings']
    out['building.footprint_area'] = out['shape.area.sum']
    out['building.floor_area'] = out['shape.floorarea.sum']
    out['building.volume'] = out['shape.volume.sum']
    out['snapshot.coverage'] = _safe_divide(
        out['building.footprint_area'], out['parcel.area']
    )
    out['snapshot.far'] = _safe_divide(
        out['building.floor_area'], out['parcel.area']
    )

    assessor_bridge = build_site_ain_bridge(aligned_parcels, assessor, year)
    assessor_metrics = _aggregate_assessor_by_site(assessor_bridge)
    out = out.reset_index().merge(
        assessor_metrics, on='site_id', how='left', validate='one_to_one'
    )
    out = gpd.GeoDataFrame(out, geometry='geometry', crs=aggregates.crs)
    out['assessor_year'] = year
    out['assessor.far'] = _safe_divide(
        out['assessor.sqft'], out['parcel.area']
    )
    out['comparison.floor_area_residual'] = (
        out['building.floor_area'] - out['assessor.sqft']
    )
    out['comparison.floor_area_ratio'] = _safe_divide(
        out['building.floor_area'], out['assessor.sqft']
    )
    out['comparison.far_residual'] = (
        out['snapshot.far'] - out['assessor.far']
    )

    # Backward-compatible aliases used by earlier exploratory notebooks.
    out['sp.bar'] = out['snapshot.coverage']
    out['sp.far'] = out['snapshot.far']
    out['sp.resid.floorarea'] = out['comparison.floor_area_residual']
    out['sp.ratio.floorarea'] = out['comparison.floor_area_ratio']

    numeric = out.select_dtypes(include='number').columns
    out[numeric] = out[numeric].round(2)
    if out.duplicated(['site_id', 'snapshot_year']).any():
        raise RuntimeError('Snapshot output is not unique by site_id and year')
    out.attrs['site_ain_bridge'] = assessor_bridge
    return out


def build_site_ain_bridge(
    parcels: gpd.GeoDataFrame,
    assessor: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    """Link assessor AIN records to physical parcel sites.

    Exact AIN matches are preferred. Remaining assessor records with point
    geometry are assigned to the containing site, which supports condominium
    unit AINs located on landbase polygons whose BPP/AIN is missing.
    """
    sites = _prepare_snapshot_parcels(parcels)
    prepared = _prepare_assessor_snapshot(assessor, year)
    assignments = prepared[['AIN']].copy()
    direct = sites.loc[sites['AIN'].notna(), ['site_id', 'AIN']]
    if direct['AIN'].duplicated().any():
        raise ValueError('One canonical AIN maps to multiple parcel sites')
    assignments = assignments.merge(
        direct, on='AIN', how='left', validate='one_to_one'
    )
    assignments['assessor.match_method'] = assignments['site_id'].notna().map(
        {True: 'AIN', False: 'unmatched'}
    )

    geometry_lookup = _assessor_geometry_by_ain(assessor)
    needs_spatial = assignments['site_id'].isna()
    if needs_spatial.any() and geometry_lookup is not None:
        candidates = assignments.loc[needs_spatial, ['AIN']].merge(
            geometry_lookup, on='AIN', how='inner', validate='one_to_one'
        )
        candidates = gpd.GeoDataFrame(
            candidates, geometry='geometry', crs=geometry_lookup.crs
        ).to_crs(sites.crs)
        present = pd.Series(
            ~pd.isna(candidates.geometry.array), index=candidates.index
        ) & ~candidates.geometry.is_empty
        spatial = gpd.sjoin(
            candidates.loc[present],
            sites[['site_id', 'geometry']],
            how='inner',
            predicate='within',
        )
        if not spatial.empty:
            spatial = spatial.sort_values(['AIN', 'site_id']).drop_duplicates('AIN')
            spatial_lookup = spatial.set_index('AIN')['site_id']
            matched = needs_spatial & assignments['AIN'].isin(spatial_lookup.index)
            assignments.loc[matched, 'site_id'] = assignments.loc[
                matched, 'AIN'
            ].map(spatial_lookup)
            assignments.loc[matched, 'assessor.match_method'] = 'center_point'

    bridge = prepared.merge(
        assignments, on='AIN', how='left', validate='one_to_one'
    )
    return bridge.sort_values(['site_id', 'AIN'], na_position='last').reset_index(
        drop=True
    )


def _assessor_geometry_by_ain(assessor):
    if not isinstance(assessor, gpd.GeoDataFrame) or assessor.crs is None:
        return None
    geometry = assessor.geometry.name
    lookup = assessor[['AIN', geometry]].copy()
    lookup['AIN'] = normalize_ain_series(
        lookup['AIN'], errors='coerce', name='AIN'
    )
    lookup = lookup.dropna(subset=['AIN']).drop_duplicates('AIN')
    result = gpd.GeoDataFrame(lookup, geometry=geometry, crs=assessor.crs)
    if geometry != 'geometry':
        result = result.rename_geometry('geometry')
    return result


def _aggregate_assessor_by_site(bridge: pd.DataFrame) -> pd.DataFrame:
    matched = bridge.dropna(subset=['site_id']).copy()
    columns = ['site_id', 'assessor.ain_count', *ASSESSOR_FIELD_NAMES.values()]
    if matched.empty:
        return pd.DataFrame(columns=columns)
    sums = [
        'assessor.sqft', 'assessor.units', 'assessor.bedrooms',
        'assessor.bathrooms', 'assessor.building_records',
    ]
    aggregations = {
        column: 'sum' for column in sums if column in matched
    }
    if 'assessor.year_built' in matched:
        aggregations['assessor.year_built'] = 'min'
    if 'assessor.effective_year_built' in matched:
        aggregations['assessor.effective_year_built'] = 'max'
    result = matched.groupby('site_id', as_index=False).agg(aggregations)
    counts = matched.groupby('site_id')['AIN'].nunique()
    result.insert(
        1, 'assessor.ain_count', result['site_id'].map(counts).astype('int64')
    )
    return result


def _prepare_assessor_snapshot(
    assessor: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    if 'AIN' not in assessor:
        raise ValueError('Assessor data are missing AIN')
    assessor = assessor.copy()
    assessor['AIN'] = normalize_ain_series(assessor['AIN'], name='AIN')
    assessor = assessor.dropna(subset=['AIN'])
    if assessor['AIN'].duplicated().any():
        raise ValueError('Assessor data must contain one row per AIN and roll')

    case_lookup = {column.lower(): column for column in assessor.columns}
    sqft_column = case_lookup.get('sqftmain')
    if sqft_column is None:
        raise ValueError('Assessor data are missing SQFTmain')
    selected = pd.DataFrame({'AIN': assessor['AIN']})
    for source_name, output_name in ASSESSOR_FIELD_NAMES.items():
        source = case_lookup.get(source_name.lower())
        selected[output_name] = (
            pd.to_numeric(assessor[source], errors='coerce')
            if source is not None else np.nan
        )
    selected['assessor_year'] = year
    return selected


def _validate_snapshot_year(frame, column, year, label):
    if column not in frame:
        return
    observed = set(pd.to_numeric(frame[column], errors='coerce').dropna().astype(int))
    if observed and observed != {year}:
        raise ValueError(
            f'{label} year does not match snapshot year {year}: {sorted(observed)}'
        )


def get_parcel_lvl_feats(
    data: gpd.GeoDataFrame,
    parcels: gpd.GeoDataFrame,
    CRS,
) -> gpd.GeoDataFrame:
    """Compatibility wrapper for spatial-only parcel aggregation."""
    out, _ = unary_join_to_parcels(
        data, parcels, CRS, PARCEL_AGGREGATE_FEATURES
    )
    out['parcel.area'] = out.geometry.area
    out['sp.bar'] = _safe_divide(out['shape.area.sum'], out['parcel.area'])
    out['sp.far'] = _safe_divide(out['shape.floorarea.sum'], out['parcel.area'])
    return out


def extract_building_to_parcel_feats(
    data: gpd.GeoDataFrame,
    parcel_metrics: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """Compatibility helper using total—not maximum—building measurements."""
    out = data.copy()
    for column in parcel_metrics:
        if column != getattr(parcel_metrics, '_geometry_column_name', None):
            out[column] = parcel_metrics[column].to_numpy()
    out['sp.bar'] = _safe_divide(out['shape.area.sum'], out['parcels.area'])
    out['sp.far'] = _safe_divide(
        out['shape.floorarea.sum'], out['parcels.area']
    )
    out['sp.resid.floorarea'] = (
        out['shape.floorarea.sum'] - out['parcels.SQFTmain']
    )
    out['sp.resid.area'] = out['shape.area.sum'] - out['parcels.SQFTmain']
    out['sp.ratio.floorarea'] = _safe_divide(
        out['shape.floorarea.sum'], out['parcels.SQFTmain']
    )
    return out


def extract_assessor_metrics(parcels: gpd.GeoDataFrame) -> pd.DataFrame:
    """Compatibility helper for assessor fields already attached to parcels."""
    metrics = _prepare_assessor_snapshot(parcels, year=0)
    metrics = metrics.rename(
        columns={
            'assessor.sqft': 'parcels.SQFTmain',
            'assessor.units': 'parcels.unitcount',
            'assessor.bedrooms': 'parcels.bdrmcount',
            'assessor.bathrooms': 'parcels.bathcount',
        }
    )
    metrics['parcels.area'] = parcels.geometry.area.to_numpy()
    metrics['parcels.far'] = _safe_divide(
        metrics['parcels.SQFTmain'], metrics['parcels.area']
    )
    return metrics

def extract_building_features(building: gpd.GeoSeries) -> dict:
    """Calculate features for one building (compatibility wrapper)."""
    row = building.to_dict()
    shape = row.get('geometry')
    if shape is None or shape.is_empty or shape.area == 0:
        return {}

    one_building = gpd.GeoDataFrame([row], geometry='geometry')
    enriched = get_building_lvl_feats(one_building)
    return enriched.iloc[0][BUILDING_FEATURE_COLUMNS].to_dict()


def extract_buildings_features(building: gpd.GeoSeries) -> dict:
    """Deprecated plural alias for :func:`extract_building_features`."""
    return extract_building_features(building)

###supporting functions

from shapely import Geometry, MultiLineString, MultiPoint, MultiPolygon, Polygon
from numpy.typing import NDArray
import math
from shapely.geometry import LineString, Point

def exchange_2d(shape: Geometry) -> float:
    radius = (shape.area / math.pi) ** 0.5
    circle = shape.centroid.buffer(radius)
    try:
        # Intersection with an invalid polygon might lead to unexpected results or errors in some cases
        # though often Shapely just returns an empty geometry for non-overlapping invalid shapes
        inter = shape.intersection(circle)
    except Exception as e:
        print(f"Error during intersection with invalid polygon: {e}")
        return 0.0
    return inter.area / shape.area

def spin_2d(shape: Geometry, raster: NDArray) -> float:
    center = raster.mean(0)
    delta = raster - center
    radius2 = shape.area / math.pi
    dist2 = (delta * delta).sum(axis=1).mean()
    return 0.5 * radius2 / dist2


def _find_a_raster(shape: Geometry, density: float) -> NDArray:
    for _ in range(5):
        raster = raster_shape(shape, density)
        if raster.shape[0] > 1:
            return raster

        density /= 2

    raise RuntimeError
def raster_shape(shape: Geometry, density=1.0) -> NDArray:
    xmin, ymin, xmax, ymax = shape.bounds
    points = []
    for x in grid_range(xmin, xmax, density):
        for y in grid_range(ymin, ymax, density):
            pt = shapely.Point(x, y)
            if shape.contains(pt):
                points.append([x, y])

    return np.array(points)
