"""Map AIN-level metrics as hexagons of assessor center points."""

import sqlite3
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
from pyproj import Transformer
from pyproj.exceptions import CRSError

try:
    from .identifiers import normalize_ain_series
except ImportError:  # Support importing modules directly from the src directory.
    from identifiers import normalize_ain_series


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MASTER_DB = PROJECT_ROOT / 'data/01_source/all_parcel_data.sqlite'
AIN_POINT_CRS = 'EPSG:2229'  # feet, so hexagon width is in ft

_ain_points_cache = {}


def _wgs84_to_point_crs():
    try:
        return Transformer.from_crs('EPSG:4326', AIN_POINT_CRS, always_xy=True)
    except CRSError:
        # A stale PROJ_LIB/PROJ_DATA from another install points pyproj at an
        # incompatible proj.db; fall back to the active environment's own copy.
        env_proj = Path(sys.prefix) / 'share' / 'proj'
        if not (env_proj / 'proj.db').exists():
            raise
        pyproj.datadir.set_data_dir(str(env_proj))
        return Transformer.from_crs('EPSG:4326', AIN_POINT_CRS, always_xy=True)


def load_ain_points(
    point_year=2016,
    fallback_year=2024,
    master_db=DEFAULT_MASTER_DB,
) -> pd.DataFrame:
    """Return one projected center point per AIN from all_parcel_data.

    Coordinates come from the ``point_year`` roll, falling back to
    ``fallback_year`` when that roll has no center. Both coordinates always come
    from the same roll. Results are cached per database and year pair.
    """
    key = (str(master_db), point_year, fallback_year)
    if key not in _ain_points_cache:
        with sqlite3.connect(master_db) as connection:
            points = pd.read_sql_query(
                f'''
                SELECT AssessorID,
                       center_lon_{point_year} AS lon, center_lat_{point_year} AS lat,
                       center_lon_{fallback_year} AS lon_fallback,
                       center_lat_{fallback_year} AS lat_fallback
                FROM all_parcel_data
                ''',
                connection,
            )
        use_fallback = points['lon'].isna() | points['lat'].isna()
        points.loc[use_fallback, ['lon', 'lat']] = (
            points.loc[use_fallback, ['lon_fallback', 'lat_fallback']].to_numpy()
        )
        points['AIN'] = normalize_ain_series(
            points['AssessorID'], errors='coerce', name='AssessorID'
        )
        points = points.dropna(subset=['AIN', 'lon', 'lat']).drop_duplicates('AIN')
        transformer = _wgs84_to_point_crs()
        points['x'], points['y'] = transformer.transform(
            points['lon'].to_numpy(), points['lat'].to_numpy()
        )
        _ain_points_cache[key] = points[['AIN', 'x', 'y']].reset_index(drop=True)
    return _ain_points_cache[key]


def plot_ain_hexbin(
    data,
    metric,
    *,
    ain_col='AIN',
    points=None,
    hex_width_ft=2000,
    reduce_function=np.median,
    min_count=5,
    cmap='viridis',
    vmin=None,
    vmax=None,
    title=None,
    colorbar_label=None,
    ax=None,
    point_year=2016,
    master_db=DEFAULT_MASTER_DB,
):
    """Map an AIN-level metric as hexagons of AIN center points.

    ``data`` needs an AIN column and a numeric or boolean ``metric`` column.
    Each hexagon shows ``reduce_function`` of the metric over its AINs; use
    ``np.mean`` on a boolean metric to map the flagged share. ``points`` may be
    a frame with ``AIN``, ``x`` and ``y`` in EPSG:2229; by default it is loaded
    from ``master_db`` with :func:`load_ain_points`.
    """
    missing = {ain_col, metric}.difference(data.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    if points is None:
        points = load_ain_points(point_year, master_db=master_db)

    frame = pd.DataFrame({
        'AIN': normalize_ain_series(data[ain_col], errors='coerce', name=ain_col),
        'value': data[metric].astype('Float64').to_numpy(
            dtype='float64', na_value=np.nan
        ),
    })
    frame = frame.merge(points, on='AIN', how='left', validate='many_to_one')
    has_point = frame['x'].notna()
    plotted = frame.loc[has_point & frame['value'].notna()]
    if plotted.empty:
        raise ValueError('No AINs with both a metric value and a center point')

    # Hexagons are only regular when the hexbin extent is square in data units.
    xmin, xmax = plotted['x'].min(), plotted['x'].max()
    ymin, ymax = plotted['y'].min(), plotted['y'].max()
    side = max(xmax - xmin, ymax - ymin, hex_width_ft)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    extent = (cx - side / 2, cx + side / 2, cy - side / 2, cy + side / 2)

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 11), constrained_layout=True)
    hexes = ax.hexbin(
        plotted['x'], plotted['y'], C=plotted['value'],
        reduce_C_function=reduce_function,
        gridsize=max(1, int(np.ceil(side / hex_width_ft))),
        extent=extent, mincnt=min_count,
        cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0,
    )
    pad = hex_width_ft
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.figure.colorbar(
        hexes, ax=ax, shrink=0.7,
        label=colorbar_label or f'{reduce_function.__name__} {metric}',
    )
    ax.set_title(title or metric)
    ax.text(
        0.0, 0.0,
        f'{len(plotted):,} AINs mapped; {int((~has_point).sum()):,} without a center point; '
        f'{hex_width_ft:,.0f} ft hexagons with ≥{min_count} AINs',
        transform=ax.transAxes, fontsize=9, color='#525252', va='top',
    )
    return ax
