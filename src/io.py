import os
import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd
import shapely
from tqdm import tqdm

try:
    from .identifiers import normalize_ain_series
    from .sites import PARCEL_SITE_COLUMNS, add_parcel_site_ids
except ImportError:  # Support importing modules directly from the src directory.
    from identifiers import normalize_ain_series
    from sites import PARCEL_SITE_COLUMNS, add_parcel_site_ids


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_CONFIG = PROJECT_ROOT / 'data/01_source/external_fp.txt'
DEFAULT_ASSESSOR_FILENAME = 'AssessorParcelsData_MultRolls.db'
DEFAULT_ASSESSOR_TABLE = 'parcels'
ASSESSOR_ROLL_YEARS = tuple(range(2006, 2025))

LARIAC_SOURCES = {
    2014: ('LARIAC4_Buildings_2014.gdb', 'LARIAC4_BUILDINGS_2014'),
    2017: ('LARIAC5_Buildings_2017.gdb', 'LARIAC5_BUILDINGS_2017'),
    2020: ('LARIAC6_Buildings_2020.gdb', 'LARIAC6_BUILDINGS_2020'),
    2023: ('LARIAC7_Buildings_2023_DINS.gpkg', 'LARIAC7_BUILDINGS_2023_DINS'),
}
LARIAC_SOURCE_COVERAGE = {
    2014: 'countywide',
    2017: 'countywide',
    2020: 'countywide',
    # The currently registered DINS layer contains only Eaton/Palisades areas.
    2023: 'partial_fire_areas',
}

PARCEL_LAYERS = {
    2011: 'Landbase2011',
    2012: 'Landbase2012',
    2013: 'Landbase2013',
    2016: 'Landbase2016',
    2017: 'Landbase2017',
    2018: 'Landbase2018',
    2021: 'Landbase_Dec2021',
    2022: 'Landbase_Dec2022',
    2023: 'Landbase_Dec2023',
    2024: 'Landbase_Dec2024',
    2025: 'LANDBASE_June2025',
}

LARIAC_COLUMN_MAP = {
    'code': 'CODE',
    'building_id': 'BLD_ID',
    'height': 'HEIGHT',
    'elevation': 'ELEV',
    'source': 'SOURCE',
    'capture_date': 'DATE_',
    'status': 'STATUS',
    'old_building_id': 'OLD_BLD_ID',
}
DEFAULT_LARIAC_COLUMNS = ('building_id', 'height', 'elevation', 'status')


def build_assessor_cache(
    output_path,
    *,
    roll_years,
    columns,
    aoi,
    parcel_ains=(),
    source_db=None,
    table_name=DEFAULT_ASSESSOR_TABLE,
    chunksize=100_000,
    overwrite=False,
):
    """Build an indexed, AOI-specific assessor cache in one source scan.

    Rows are retained when their assessor center lies inside ``aoi`` or their
    AIN belongs to an AOI parcel. The latter protects parcels near the city
    boundary and landbase/assessor center disagreements. Only requested roll
    years and columns are copied.
    """
    output_path = Path(output_path)
    years = tuple(sorted({int(year) for year in roll_years}))
    if not years:
        raise ValueError('roll_years must contain at least one year')
    unsupported = sorted(set(years).difference(ASSESSOR_ROLL_YEARS))
    if unsupported:
        raise ValueError(f'Unsupported assessor roll years: {unsupported}')
    if chunksize <= 0:
        raise ValueError('chunksize must be positive')
    if source_db is None:
        registry = read_external_source_paths()
        source_db = registry.get('assessor database')
        if source_db is None:
            raise ValueError('Assessor database is missing from external_fp.txt')
        source_db = Path(source_db)
        if source_db.is_dir():
            source_db = source_db / DEFAULT_ASSESSOR_FILENAME
    source_db = Path(source_db)
    if not source_db.exists():
        raise FileNotFoundError(source_db)
    requested = list(dict.fromkeys([
        'AIN', 'AssessorID', 'RollYear', *columns, 'CENTER_LON', 'CENTER_LAT'
    ]))
    if output_path.exists() and not overwrite:
        _validate_assessor_cache(output_path, years, requested)
        return output_path

    polygon = _polygon_in_wgs84(aoi, 'EPSG:4326')
    if polygon is None or polygon.is_empty or not polygon.is_valid:
        raise ValueError('aoi must contain one valid, non-empty geometry')
    shapely.prepare(polygon)
    known_ains = {
        int(value) for value in normalize_ain_series(
            pd.Series(list(parcel_ains)), errors='coerce'
        ).dropna()
    }
    source_table = _quote_identifier(table_name)
    placeholders = ','.join('?' for _ in years)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + '.tmp')
    if temporary.exists():
        temporary.unlink()
    wrote_rows = False
    total_rows = 0
    try:
        with sqlite3.connect(source_db) as source, sqlite3.connect(temporary) as target:
            available = {
                row[1] for row in source.execute(f'PRAGMA table_info({source_table})')
            }
            missing = sorted(set(requested).difference(available))
            if missing:
                raise ValueError(f'Columns not found in assessor table: {missing}')
            selected_sql = ', '.join(_quote_identifier(column) for column in requested)
            query = (
                f'SELECT {selected_sql} FROM {source_table} '
                f'WHERE "RollYear" IN ({placeholders})'
            )
            reader = pd.read_sql_query(
                query, source, params=years, chunksize=chunksize
            )
            for chunk in tqdm(reader, desc='Building assessor cache'):
                ain_values = pd.to_numeric(chunk['AIN'], errors='coerce')
                ain_mask = ain_values.isin(known_ains)
                coordinate_mask = chunk['CENTER_LON'].notna() & chunk['CENTER_LAT'].notna()
                spatial_mask = pd.Series(False, index=chunk.index)
                if coordinate_mask.any():
                    points = gpd.points_from_xy(
                        chunk.loc[coordinate_mask, 'CENTER_LON'],
                        chunk.loc[coordinate_mask, 'CENTER_LAT'],
                    )
                    spatial_mask.loc[coordinate_mask] = shapely.contains(
                        polygon, points
                    )
                retained = chunk.loc[ain_mask | spatial_mask]
                if retained.empty:
                    continue
                retained.to_sql(
                    table_name,
                    target,
                    if_exists='append' if wrote_rows else 'replace',
                    index=False,
                )
                wrote_rows = True
                total_rows += len(retained)
            if not wrote_rows:
                pd.DataFrame(columns=requested).to_sql(
                    table_name, target, if_exists='replace', index=False
                )
            target.execute(
                f'CREATE INDEX idx_{table_name}_roll_ain '
                f'ON {_quote_identifier(table_name)} ("RollYear", "AIN")'
            )
            target.execute(
                f'CREATE INDEX idx_{table_name}_roll_coordinates '
                f'ON {_quote_identifier(table_name)} '
                '("RollYear", "CENTER_LON", "CENTER_LAT")'
            )
            metadata = pd.DataFrame([{
                'roll_years': json.dumps(years),
                'columns': json.dumps(requested),
                'row_count': int(total_rows),
                'source_db': str(source_db),
            }])
            metadata.to_sql('cache_metadata', target, if_exists='replace', index=False)
            target.execute('ANALYZE')
        if output_path.exists():
            output_path.unlink()
        temporary.rename(output_path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return output_path


def _validate_assessor_cache(path, years, columns):
    try:
        with sqlite3.connect(path) as connection:
            row = connection.execute(
                'SELECT roll_years, columns FROM cache_metadata LIMIT 1'
            ).fetchone()
    except sqlite3.Error as error:
        raise ValueError(f'Invalid assessor cache: {path}') from error
    if row is None:
        raise ValueError(f'Assessor cache has no metadata: {path}')
    cached_years = tuple(json.loads(row[0]))
    cached_columns = set(json.loads(row[1]))
    if cached_years != tuple(years) or not set(columns).issubset(cached_columns):
        raise ValueError(
            f'Assessor cache parameters do not match this run: {path}'
        )


def read_external_source_paths(config_path=DEFAULT_SOURCE_CONFIG) -> dict[str, Path]:
    """Read the small ``label: path`` registry for external source data."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    paths = {}
    for line in config_path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#') or ':' not in line:
            continue
        label, value = line.split(':', 1)
        paths[label.strip().lower()] = Path(value.strip()).expanduser()
    return paths


def load_lariac(
    year: int,
    aoi=None,
    *,
    columns: Optional[Sequence[str]] = None,
    source_dir=None,
    target_crs=None,
    aoi_crs='EPSG:4326',
    building_only=True,
) -> gpd.GeoDataFrame:
    """Load one LARIAC vintage with a stable schema and optional AOI filter.

    ``columns`` uses canonical names from :data:`LARIAC_COLUMN_MAP`. Geometry,
    ``code``, and ``lariac_year`` are supplied automatically. The AOI may be a
    GeoDataFrame, GeoSeries, Shapely geometry, or four-value bounding box.
    """
    year = int(year)
    if year not in LARIAC_SOURCES:
        raise ValueError(
            f'Unsupported LARIAC year {year}; choose from {sorted(LARIAC_SOURCES)}'
        )
    if source_dir is None:
        registry = read_external_source_paths()
        source_dir = registry.get('lariac footprint directory')
        if source_dir is None:
            raise ValueError('LARIAC directory is missing from external_fp.txt')
    filename, layer = LARIAC_SOURCES[year]
    source = Path(source_dir) / filename
    if not source.exists():
        raise FileNotFoundError(source)

    requested = list(DEFAULT_LARIAC_COLUMNS if columns is None else columns)
    unknown = sorted(set(requested).difference(LARIAC_COLUMN_MAP))
    if unknown:
        raise ValueError(f'Unknown canonical LARIAC columns: {unknown}')
    canonical = list(dict.fromkeys(['code', *requested]))
    raw_columns = [LARIAC_COLUMN_MAP[column] for column in canonical]
    source_aoi = _aoi_for_source(aoi, source, layer, aoi_crs)
    buildings = gpd.read_file(
        source,
        layer=layer,
        columns=raw_columns,
        bbox=None if source_aoi is None else tuple(source_aoi.total_bounds),
    )
    buildings = buildings.rename(
        columns={raw: name for name, raw in LARIAC_COLUMN_MAP.items()}
    )
    if source_aoi is not None:
        buildings = buildings[buildings.intersects(source_aoi.iloc[0])]
    if building_only:
        buildings = buildings[buildings['code'].eq('Building')]
    buildings = buildings.reset_index(drop=True)
    buildings['lariac_year'] = year
    if target_crs is not None and buildings.crs != target_crs:
        buildings = buildings.to_crs(target_crs)
    return buildings


def load_parcel_polygons(
    year: int,
    aoi=None,
    *,
    columns: Optional[Sequence[str]] = None,
    source_path=None,
    target_crs=None,
    aoi_crs='EPSG:4326',
    include_missing_ain=False,
) -> gpd.GeoDataFrame:
    """Load landbase parcel polygons for a year and normalize ``BPP`` to AIN.

    Landbase layers also contain rights-of-way and other polygons without a
    usable ``BPP``. These are excluded by default because they cannot produce
    parcel-level outputs; set ``include_missing_ain=True`` to retain them.
    """
    year = int(year)
    if year not in PARCEL_LAYERS:
        raise ValueError(
            f'Unsupported parcel year {year}; choose from {sorted(PARCEL_LAYERS)}'
        )
    if source_path is None:
        registry = read_external_source_paths()
        source_path = registry.get('parcel data landbase')
        if source_path is None:
            raise ValueError('Parcel landbase is missing from external_fp.txt')
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(source)

    requested = [
        column for column in (columns or []) if column not in {'AIN', 'BPP'}
    ]
    raw_columns = list(dict.fromkeys(['BPP', *PARCEL_SITE_COLUMNS, *requested]))
    layer = PARCEL_LAYERS[year]
    source_aoi = _aoi_for_source(aoi, source, layer, aoi_crs)
    parcels = gpd.read_file(
        source,
        layer=layer,
        columns=raw_columns,
        bbox=None if source_aoi is None else tuple(source_aoi.total_bounds),
    )
    if source_aoi is not None:
        parcels = parcels[parcels.intersects(source_aoi.iloc[0])]
    parcels = parcels.rename(columns={'BPP': 'AIN'}).reset_index(drop=True)
    source_row_count = len(parcels)
    present = pd.Series(
        ~pd.isna(parcels.geometry.array), index=parcels.index
    ) & ~parcels.geometry.is_empty
    invalid = present & ~parcels.geometry.is_valid
    if invalid.any():
        parcels.loc[invalid, parcels.geometry.name] = parcels.loc[
            invalid, parcels.geometry.name
        ].map(_polygonal_make_valid)
    retained_geometry = pd.Series(
        ~pd.isna(parcels.geometry.array), index=parcels.index
    ) & ~parcels.geometry.is_empty & parcels.geometry.geom_type.isin(
        ['Polygon', 'MultiPolygon']
    )
    dropped_geometry_count = int((~retained_geometry).sum())
    parcels = parcels.loc[retained_geometry].reset_index(drop=True)
    parcels['AIN'] = normalize_ain_series(
        parcels['AIN'], errors='coerce', name='AIN'
    )
    parcels = add_parcel_site_ids(parcels)
    missing_ain_count = int(parcels['AIN'].isna().sum())
    if not include_missing_ain:
        parcels = parcels.loc[parcels['AIN'].notna()].reset_index(drop=True)
    parcels['parcel_year'] = year
    if target_crs is not None and parcels.crs != target_crs:
        parcels = parcels.to_crs(target_crs)
    parcels.attrs['loading_diagnostics'] = {
        'source_rows': int(source_row_count),
        'invalid_geometries_repaired': int(invalid.sum()),
        'non_polygon_or_empty_geometries_dropped': dropped_geometry_count,
        'missing_ain_rows': missing_ain_count,
        'returned_rows': int(len(parcels)),
        'included_missing_ain': bool(include_missing_ain),
        'returned_sites': int(parcels['site_id'].nunique()),
    }
    return parcels


def _polygonal_make_valid(geometry):
    """Return the polygonal portion of a repaired source geometry."""
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


def load_assessor_columns(
    roll_year: int,
    columns: Sequence[str],
    *,
    ains=None,
    aoi=None,
    db_filepath=None,
    table_name=DEFAULT_ASSESSOR_TABLE,
    aoi_crs='EPSG:4326',
) -> pd.DataFrame:
    """Load one annual assessor roll, restricted by AINs or a spatial AOI.

    At least one restriction is required so an accidental call cannot load the
    entire multi-roll table. Spatial results are returned as a GeoDataFrame in
    EPSG:4326; AIN-only results are a regular DataFrame.
    """
    if isinstance(columns, str):
        columns = [columns]
    else:
        columns = list(columns)
    if ains is None and aoi is None:
        raise ValueError('Provide ains or aoi to restrict the assessor query')

    roll_year = int(roll_year)
    if roll_year not in ASSESSOR_ROLL_YEARS:
        raise ValueError(
            f'Unsupported assessor roll {roll_year}; choose from '
            f'{ASSESSOR_ROLL_YEARS[0]} through {ASSESSOR_ROLL_YEARS[-1]}'
        )
    if db_filepath is None:
        registry = read_external_source_paths()
        assessor_source = registry.get('assessor database')
        if assessor_source is None:
            raise ValueError('Assessor database is missing from external_fp.txt')
        db_filepath = (
            assessor_source / DEFAULT_ASSESSOR_FILENAME
            if assessor_source.is_dir()
            else assessor_source
        )
    db_filepath = Path(db_filepath)
    if not db_filepath.exists():
        raise FileNotFoundError(db_filepath)
    id_column = 'AssessorID'
    selected = list(dict.fromkeys([id_column, 'RollYear', *columns]))

    if aoi is not None:
        result = query_points_within_geometry(
            db_filepath,
            table_name,
            aoi,
            lon_col='CENTER_LON',
            lat_col='CENTER_LAT',
            columns=selected,
            polygon_crs=aoi_crs,
            filters={'RollYear': roll_year},
        )
    else:
        result = _query_assessor_by_ain(
            db_filepath, table_name, selected, ains, roll_year
        )

    result = result.rename(
        columns={id_column: 'AIN', 'RollYear': 'assessor_year'}
    )
    result['AIN'] = normalize_ain_series(
        result['AIN'], errors='coerce', name='AIN'
    )
    if ains is not None and aoi is not None:
        requested_ains = set(
            normalize_ain_series(ains, errors='coerce').dropna()
        )
        result = result[result['AIN'].isin(requested_ains)]
    return result.reset_index(drop=True)


def _query_assessor_by_ain(
    db_filepath, table_name, columns, ains, roll_year
):
    normalized = normalize_ain_series(ains, errors='coerce').dropna().unique()
    if len(normalized) == 0:
        return pd.DataFrame(columns=columns)

    table_sql = _quote_identifier(table_name)
    selected_sql = ', '.join(
        f'p.{_quote_identifier(column)}' for column in columns
    )
    with sqlite3.connect(db_filepath) as conn:
        available = {
            row[1] for row in conn.execute(f'PRAGMA table_info({table_sql})')
        }
        missing = sorted(set(columns).difference(available))
        if missing:
            raise ValueError(f'Columns not found in table: {missing}')
        conn.execute(
            'CREATE TEMP TABLE requested_ains '
            '(AIN INTEGER PRIMARY KEY) WITHOUT ROWID'
        )
        conn.executemany(
            'INSERT INTO requested_ains (AIN) VALUES (?)',
            ((int(value),) for value in normalized),
        )
        sql = f'''
            SELECT {selected_sql}
            FROM {table_sql} AS p
            INNER JOIN requested_ains AS requested ON requested.AIN = p.AIN
            WHERE p.RollYear = ?
        '''
        return pd.read_sql_query(sql, conn, params=(roll_year,))


def _aoi_for_source(aoi, source, layer, aoi_crs):
    if aoi is None:
        return None
    if isinstance(aoi, (tuple, list)) and len(aoi) == 4:
        aoi = shapely.box(*aoi)
    if isinstance(aoi, gpd.GeoDataFrame):
        geometries = aoi.geometry
    elif isinstance(aoi, gpd.GeoSeries):
        geometries = aoi
    else:
        geometries = gpd.GeoSeries([aoi], crs=aoi_crs)
    if geometries.crs is None:
        geometries = geometries.set_crs(aoi_crs)
    source_crs = gpd.read_file(source, layer=layer, rows=0).crs
    if source_crs is None:
        raise ValueError(f'Source layer {layer!r} has no CRS')
    geometry = geometries.to_crs(source_crs).union_all()
    if geometry.is_empty or not geometry.is_valid:
        raise ValueError('AOI must contain one valid, non-empty geometry')
    return gpd.GeoSeries([geometry], crs=source_crs)

def csv_to_sqlite(csv_filepath, db_filepath, table_name, chunksize=100_000,
                  encodings=['utf-8', 'cp1252'], sep=',', dtype_mapping=None,
                  on_bad_lines='warn'):
    """
    Transfers a large CSV to SQLite in chunks with automatic encoding fallback.
    """

    if not os.path.exists(csv_filepath):
        print(f"Error: File '{csv_filepath}' not found.")
        return

    # Ensure encodings is a list for iteration
    if isinstance(encodings, str):
        encodings = [encodings]

    for enc in encodings:
        try:
            print(f"Attempting to read '{csv_filepath}' with encoding: {enc}")

            # Context manager handles closing the connection automatically
            with sqlite3.connect(db_filepath) as conn:
                reader = pd.read_csv(
                    csv_filepath,
                    chunksize=chunksize,
                    encoding=enc,
                    sep=sep,
                    dtype=dtype_mapping,
                    on_bad_lines=on_bad_lines,
                    engine='c'  # 'c' is faster than 'python' for standard CSVs
                )

                for i, chunk in tqdm(enumerate(reader), desc='Processing csv chunk:'):
                    # Replace table on the first chunk, append on subsequent chunks
                    mode = 'replace' if i == 0 else 'append'

                    chunk.to_sql(table_name, conn, if_exists=mode, index=False)

                    # # Optional: Print progress every 5 chunks
                    # if (i + 1) % 5 == 0:
                    #     print(f"  Processed approx. {(i + 1) * chunksize:,} rows...")

            print(f"Success! Data saved to table '{table_name}' in '{db_filepath}'.")
            return  # Exit function immediately on success

        except UnicodeDecodeError:
            print(f"  Encoding '{enc}' failed. Retrying with next encoding...")
            continue  # Move to the next encoding in the list
        except Exception as e:
            print(f"  Failed with critical error: {e}")
            return  # Stop execution on non-encoding errors

    print("Error: Failed to read CSV with all provided encodings.")


def query_points_within_geometry(
    db_filepath,
    table_name,
    polygon,
    *,
    lon_col='center_lon_2024',
    lat_col='center_lat_2024',
    columns=None,
    polygon_crs=None,
    include_boundary=False,
    chunksize=50_000,
    filters=None,
):
    """Query coordinate rows that fall inside a polygon.

    SQLite first applies an indexed bounding-box filter. GeoPandas then applies
    the exact point-in-polygon predicate to each result chunk. Coordinates in
    the database are assumed to use EPSG:4326.

    ``polygon`` may be a Shapely geometry, GeoSeries, or GeoDataFrame. A plain
    Shapely geometry requires ``polygon_crs`` unless it is already EPSG:4326.
    Use ``columns`` to avoid loading all columns from a very wide table.
    """
    polygon_geometry = _polygon_in_wgs84(polygon, polygon_crs)
    if polygon_geometry is None or polygon_geometry.is_empty:
        raise ValueError('polygon must contain one non-empty geometry')
    if not polygon_geometry.is_valid:
        raise ValueError('polygon geometry is invalid')
    if chunksize <= 0:
        raise ValueError('chunksize must be positive')

    # Reusing one prepared polygon makes exact tests against a complex boundary
    # much faster than evaluating point.within(polygon) independently.
    shapely.prepare(polygon_geometry)

    table_sql = _quote_identifier(table_name)
    lon_sql = _quote_identifier(lon_col)
    lat_sql = _quote_identifier(lat_col)

    with sqlite3.connect(db_filepath) as conn:
        available_columns = [
            row[1] for row in conn.execute(f'PRAGMA table_info({table_sql})')
        ]
        if not available_columns:
            raise ValueError(f'Table {table_name!r} does not exist or has no columns')

        requested_columns = _select_query_columns(
            columns, available_columns, lon_col, lat_col
        )
        filters = dict(filters or {})
        missing_filters = sorted(set(filters).difference(available_columns))
        if missing_filters:
            raise ValueError(f'Filter columns not found in table: {missing_filters}')
        select_sql = ', '.join(_quote_identifier(column) for column in requested_columns)
        minx, miny, maxx, maxy = polygon_geometry.bounds
        filter_sql = ''.join(
            f' AND {_quote_identifier(column)} = ?' for column in filters
        )
        sql = f"""
            SELECT {select_sql}
            FROM {table_sql}
            WHERE {lon_sql} BETWEEN ? AND ?
              AND {lat_sql} BETWEEN ? AND ?
              AND {lon_sql} IS NOT NULL
              AND {lat_sql} IS NOT NULL
              {filter_sql}
        """

        matches = []
        reader = pd.read_sql_query(
            sql,
            conn,
            params=(minx, maxx, miny, maxy, *filters.values()),
            chunksize=chunksize,
        )
        for chunk in reader:
            points = gpd.GeoDataFrame(
                chunk,
                geometry=gpd.points_from_xy(chunk[lon_col], chunk[lat_col]),
                crs='EPSG:4326',
            )
            mask = (
                shapely.intersects(polygon_geometry, points.geometry.array)
                if include_boundary
                else shapely.contains(polygon_geometry, points.geometry.array)
            )
            if mask.any():
                matches.append(points.loc[mask].copy())

    if not matches:
        empty = pd.DataFrame(columns=requested_columns)
        return gpd.GeoDataFrame(empty, geometry=gpd.GeoSeries(crs='EPSG:4326'))

    result = pd.concat(matches, ignore_index=True)
    return gpd.GeoDataFrame(result, geometry='geometry', crs='EPSG:4326')


def _query_within_geometry(db_filepath, table_name, polygon_geometry):
    """Backward-compatible wrapper for legacy uppercase coordinate columns."""
    return query_points_within_geometry(
        db_filepath,
        table_name,
        polygon_geometry,
        lon_col='CENTER_LON',
        lat_col='CENTER_LAT',
        polygon_crs='EPSG:4326',
    )


def _polygon_in_wgs84(polygon, polygon_crs):
    """Normalize a polygon-like input to one WGS84 Shapely geometry."""
    if isinstance(polygon, (tuple, list)) and len(polygon) == 4:
        polygon = shapely.box(*polygon)
    if isinstance(polygon, gpd.GeoDataFrame):
        geometries = polygon.geometry
    elif isinstance(polygon, gpd.GeoSeries):
        geometries = polygon
    else:
        source_crs = polygon_crs or 'EPSG:4326'
        geometries = gpd.GeoSeries([polygon], crs=source_crs)

    if geometries.crs is None:
        if polygon_crs is None:
            raise ValueError('polygon CRS is missing; provide polygon_crs')
        geometries = geometries.set_crs(polygon_crs)
    if geometries.crs != 'EPSG:4326':
        geometries = geometries.to_crs('EPSG:4326')
    present = pd.Series(
        ~pd.isna(geometries.array), index=geometries.index
    ) & ~geometries.is_empty
    geometries = geometries[present]
    if geometries.empty:
        return None
    return geometries.union_all()


def _select_query_columns(
    columns: Optional[Sequence[str]],
    available_columns: Sequence[str],
    lon_col: str,
    lat_col: str,
) -> list[str]:
    """Validate requested fields and include coordinates needed for geometry."""
    if columns is None:
        selected = list(available_columns)
    elif isinstance(columns, str):
        selected = [columns]
    else:
        selected = list(columns)

    for coordinate in (lon_col, lat_col):
        if coordinate not in selected:
            selected.append(coordinate)
    missing = sorted(set(selected).difference(set(available_columns)))
    if missing:
        raise ValueError(f'Columns not found in table: {missing}')
    return selected


def _quote_identifier(identifier: str) -> str:
    """Safely quote a SQLite table or column identifier."""
    if not isinstance(identifier, str) or not identifier:
        raise ValueError('SQLite identifiers must be non-empty strings')
    return '"' + identifier.replace('"', '""') + '"'
