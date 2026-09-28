import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, box

from src.io import (
    LARIAC_SOURCES,
    build_assessor_cache,
    load_assessor_columns,
    load_lariac,
    load_parcel_polygons,
    query_points_within_geometry,
)


class StandardizedLoadingTests(unittest.TestCase):
    def test_builds_single_roll_indexed_assessor_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory, 'source.sqlite')
            cache = Path(directory, 'cache.sqlite')
            records = pd.DataFrame({
                'AIN': [1, 2, 3, 4],
                'AssessorID': ['0000-000-001', '0000-000-002', '0000-000-003', '0000-000-004'],
                'RollYear': [2017, 2017, 2017, 2020],
                'SQFTmain': [100, 200, 300, 400],
                'CENTER_LON': [-118.40, -117.00, -117.10, -118.40],
                'CENTER_LAT': [34.00, 35.00, 35.10, 34.00],
            })
            with sqlite3.connect(source) as connection:
                records.to_sql('parcels', connection, index=False)

            build_assessor_cache(
                cache,
                roll_years=[2017],
                columns=['SQFTmain'],
                aoi=(-118.5, 33.9, -118.3, 34.1),
                parcel_ains=['0000000002'],
                source_db=source,
                chunksize=2,
            )
            with sqlite3.connect(cache) as connection:
                cached = pd.read_sql_query(
                    'SELECT AIN, RollYear FROM parcels ORDER BY AIN', connection
                )
                indexes = {
                    row[1] for row in connection.execute("PRAGMA index_list('parcels')")
                }

        self.assertEqual(cached['AIN'].tolist(), [1, 2])
        self.assertTrue(cached['RollYear'].eq(2017).all())
        self.assertEqual(indexes, {
            'idx_parcels_roll_ain', 'idx_parcels_roll_coordinates'
        })

    def test_lariac_loader_returns_canonical_columns(self):
        source = gpd.GeoDataFrame(
            {
                'CODE': ['Building', 'Other'],
                'BLD_ID': [10, 11],
                'HEIGHT': [20.0, 5.0],
                'ELEV': [100.0, 101.0],
                'STATUS': ['Existing', 'Existing'],
            },
            geometry=[box(0, 0, 1, 1), box(2, 2, 3, 3)],
            crs='EPSG:2229',
        )
        with tempfile.TemporaryDirectory() as directory:
            filename, _ = LARIAC_SOURCES[2020]
            Path(directory, filename).touch()
            with patch('src.io.gpd.read_file', return_value=source):
                result = load_lariac(2020, source_dir=directory)

        self.assertEqual(len(result), 1)
        self.assertEqual(
            result.columns.tolist(),
            [
                'code', 'building_id', 'height', 'elevation', 'status',
                'geometry', 'lariac_year',
            ],
        )
        self.assertEqual(result.loc[0, 'lariac_year'], 2020)

    def test_parcel_loader_normalizes_ain(self):
        source = gpd.GeoDataFrame(
            {'BPP': ['2004-001-003'], 'TRACT': ['12345']},
            geometry=[box(0, 0, 10, 10)],
            crs='EPSG:2229',
        )
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory, 'landbase.gdb')
            source_path.mkdir()
            with patch('src.io.gpd.read_file', return_value=source):
                result = load_parcel_polygons(
                    2017, columns=['AIN', 'TRACT'], source_path=source_path
                )

        self.assertEqual(result.loc[0, 'AIN'], '2004001003')
        self.assertEqual(result.loc[0, 'parcel_year'], 2017)
        self.assertIn('TRACT', result)

    def test_parcel_loader_excludes_non_parcel_landbase_rows(self):
        source = gpd.GeoDataFrame(
            {'BPP': ['2004-001-003', None]},
            geometry=[box(0, 0, 10, 10), box(10, 0, 20, 10)],
            crs='EPSG:2229',
        )
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory, 'landbase.gdb')
            source_path.mkdir()
            with patch('src.io.gpd.read_file', return_value=source):
                result = load_parcel_polygons(
                    2017, source_path=source_path
                )

        self.assertEqual(result['AIN'].tolist(), ['2004001003'])
        self.assertEqual(result.attrs['loading_diagnostics'], {
            'source_rows': 2,
            'invalid_geometries_repaired': 0,
            'non_polygon_or_empty_geometries_dropped': 0,
            'missing_ain_rows': 1,
            'returned_rows': 1,
            'included_missing_ain': False,
            'returned_sites': 1,
        })

    def test_parcel_loader_repairs_invalid_polygon(self):
        invalid = Polygon([(0, 0), (10, 10), (0, 10), (10, 0), (0, 0)])
        source = gpd.GeoDataFrame(
            {'BPP': ['2004-001-003']}, geometry=[invalid], crs='EPSG:2229'
        )
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory, 'landbase.gdb')
            source_path.mkdir()
            with patch('src.io.gpd.read_file', return_value=source):
                result = load_parcel_polygons(2017, source_path=source_path)

        self.assertEqual(len(result), 1)
        self.assertTrue(result.geometry.is_valid.all())
        diagnostics = result.attrs['loading_diagnostics']
        self.assertEqual(diagnostics['invalid_geometries_repaired'], 1)
        self.assertEqual(diagnostics['non_polygon_or_empty_geometries_dropped'], 0)

    def test_assessor_loader_selects_columns_and_accepts_compact_ain(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory, 'assessor.sqlite')
            records = pd.DataFrame(
                {
                    'AIN': [2004001003, 2004001004],
                    'AssessorID': ['2004-001-003', '2004-001-004'],
                    'RollYear': [2014, 2014],
                    'SQFTmain': [2090, 2479],
                    'Units': [1, 2],
                }
            )
            with sqlite3.connect(database) as connection:
                records.to_sql('parcels', connection, index=False)
            result = load_assessor_columns(
                2014,
                ['SQFTmain'],
                ains=['2004001003'],
                db_filepath=database,
            )

        self.assertEqual(
            result.columns.tolist(),
            ['AIN', 'assessor_year', 'SQFTmain'],
        )
        self.assertEqual(result.loc[0, 'AIN'], '2004001003')
        self.assertEqual(result.loc[0, 'assessor_year'], 2014)
        self.assertEqual(result.loc[0, 'SQFTmain'], 2090)

    def test_assessor_loader_requires_a_filter(self):
        with self.assertRaisesRegex(ValueError, 'Provide ains or aoi'):
            load_assessor_columns(2014, ['SQFTmain'])

    def test_spatial_query_accepts_bbox_tuple(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory, 'points.sqlite')
            records = pd.DataFrame({
                'AIN': [1, 2],
                'lon': [-118.4, -117.0],
                'lat': [34.0, 35.0],
            })
            with sqlite3.connect(database) as connection:
                records.to_sql('points', connection, index=False)
            result = query_points_within_geometry(
                database,
                'points',
                (-118.5, 33.9, -118.3, 34.1),
                lon_col='lon',
                lat_col='lat',
                columns=['AIN'],
            )

        self.assertEqual(result['AIN'].tolist(), [1])


if __name__ == '__main__':
    unittest.main()
