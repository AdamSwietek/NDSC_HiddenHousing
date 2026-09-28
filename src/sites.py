"""Stable physical-site identifiers for parcel geometries."""

from __future__ import annotations

import hashlib

import geopandas as gpd
import pandas as pd

try:
    from .identifiers import normalize_ain_series
except ImportError:  # Support importing modules directly from the src directory.
    from identifiers import normalize_ain_series


PARCEL_SITE_COLUMNS = ('ASSETID', 'PIND', 'PIN')


def add_parcel_site_ids(parcels: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return parcels with a stable physical-site identifier.

    Canonical AINs remain the preferred identifier for ordinary parcels so
    multipart landbase records dissolve to one site. Landbase polygons without
    an AIN use ``ASSETID``, then ``PIND`` or ``PIN``. A geometry digest is the
    final deterministic fallback for older or synthetic layers.
    """
    if not isinstance(parcels, gpd.GeoDataFrame):
        raise TypeError('parcels must be a GeoDataFrame')
    if 'AIN' not in parcels:
        raise ValueError('Parcels are missing AIN')

    out = parcels.copy()
    out['AIN'] = normalize_ain_series(
        out['AIN'], errors='coerce', name='AIN'
    )
    site_id = pd.Series(pd.NA, index=out.index, dtype='string')
    has_ain = out['AIN'].notna()
    site_id.loc[has_ain] = 'ain:' + out.loc[has_ain, 'AIN']

    for column, prefix in (
        ('ASSETID', 'asset'), ('PIND', 'pind'), ('PIN', 'pin')
    ):
        if column not in out:
            continue
        values = out[column].map(_site_component).astype('string')
        usable = site_id.isna() & values.notna() & values.ne('')
        site_id.loc[usable] = prefix + ':' + values.loc[usable]

    missing = site_id.isna()
    if missing.any():
        site_id.loc[missing] = out.loc[missing].geometry.map(
            lambda geometry: 'geometry:' + hashlib.sha1(
                geometry.wkb if geometry is not None else b''
            ).hexdigest()[:20]
        )
    out['site_id'] = site_id
    return out


def _site_component(value):
    if pd.isna(value):
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()
