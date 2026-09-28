import unittest

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon, box

from src.metrics import build_snapshot_metrics


class SnapshotMetricsTests(unittest.TestCase):
    def setUp(self):
        self.parcels = gpd.GeoDataFrame(
            {
                'AIN': ['2004-001-003', '2004-001-004'],
                'parcel_year': [2017, 2017],
            },
            geometry=[box(0, 0, 10, 10), box(10, 0, 20, 10)],
            crs='EPSG:2229',
        )
        self.buildings = gpd.GeoDataFrame(
            {
                'building_id': [1, 2, 3, 4],
                'height': [20.0, 30.0, 10.0, 10.0],
                'elevation': [100.0, 100.0, 100.0, 100.0],
                'lariac_year': [2017, 2017, 2017, 2017],
            },
            geometry=[
                box(1, 1, 3, 3),
                box(4, 1, 6, 3),
                box(12, 1, 14, 3),
                box(30, 1, 32, 3),
            ],
            crs='EPSG:2229',
        )
        self.assessor = pd.DataFrame(
            {
                'AIN': ['2004-001-003', '2004-001-004'],
                'assessor_year': [2017, 2017],
                'SQFTmain': [16, 4],
                'Units': [2, 1],
                'Bedrooms': [5, 2],
                'Bathrooms': [3, 1],
                'YearBuilt': [1950, 1960],
            }
        )

    def test_builds_building_and_parcel_snapshot_outputs(self):
        buildings, parcels = build_snapshot_metrics(
            self.buildings, self.parcels, self.assessor, 2017
        )

        self.assertEqual(len(buildings), 4)
        self.assertEqual(len(parcels), 2)
        self.assertEqual(
            parcels[['AIN', 'snapshot_year']].duplicated().sum(), 0
        )
        self.assertEqual(
            buildings['AIN'].dropna().tolist(),
            ['2004001003', '2004001003', '2004001004'],
        )
        self.assertTrue(pd.isna(buildings.loc[3, 'AIN']))

    def test_uses_totals_for_coverage_far_and_assessor_comparison(self):
        _, parcels = build_snapshot_metrics(
            self.buildings, self.parcels, self.assessor, 2017
        )
        first = parcels.set_index('AIN').loc['2004001003']

        self.assertEqual(first['building.count'], 2)
        self.assertEqual(first['building.footprint_area'], 8.0)
        self.assertEqual(first['building.floor_area'], 20.0)
        self.assertEqual(first['building.volume'], 200.0)
        self.assertEqual(first['snapshot.coverage'], 0.08)
        self.assertEqual(first['snapshot.far'], 0.20)
        self.assertEqual(first['assessor.sqft'], 16)
        self.assertEqual(first['assessor.bedrooms'], 5)
        self.assertEqual(first['assessor.bathrooms'], 3)
        self.assertEqual(first['comparison.floor_area_residual'], 4.0)
        self.assertEqual(first['comparison.floor_area_ratio'], 1.25)
        self.assertEqual(first['comparison.far_residual'], 0.04)

    def test_preserves_matching_diagnostics(self):
        buildings, parcels = build_snapshot_metrics(
            self.buildings, self.parcels, self.assessor, 2017
        )
        for output in (buildings, parcels):
            diagnostics = output.attrs['matching_diagnostics']
            self.assertEqual(diagnostics['matched_buildings'], 3)
            self.assertEqual(diagnostics['unmatched_buildings'], 1)

    def test_repairs_invalid_source_building_before_matching(self):
        buildings = self.buildings.iloc[:1].copy()
        buildings.geometry = [
            Polygon([(1, 1), (4, 4), (1, 4), (4, 1), (1, 1)])
        ]
        self.assertFalse(buildings.geometry.iloc[0].is_valid)

        output, _ = build_snapshot_metrics(
            buildings,
            self.parcels,
            self.assessor,
            2017,
            repair_boundary_artifacts=False,
        )

        self.assertEqual(len(output), 1)
        self.assertTrue(output.geometry.is_valid.all())
        self.assertEqual(
            output.attrs['geometry_cleaning_diagnostics'],
            {
                'input_buildings': 1,
                'invalid_buildings_repaired': 1,
                'non_polygon_or_empty_buildings_dropped': 0,
                'output_buildings': 1,
            },
        )

    def test_rejects_mismatched_assessor_year(self):
        assessor = self.assessor.assign(assessor_year=2016)
        with self.assertRaisesRegex(ValueError, 'assessor year'):
            build_snapshot_metrics(
                self.buildings, self.parcels, assessor, 2017
            )

    def test_matches_null_ain_site_and_aggregates_condominium_ains(self):
        parcels = gpd.GeoDataFrame(
            {
                'AIN': ['2004-001-003', None],
                'ASSETID': [1, 2],
                'parcel_year': [2017, 2017],
            },
            geometry=[box(0, 0, 10, 10), box(10, 0, 20, 10)],
            crs='EPSG:2229',
        )
        buildings = self.buildings.iloc[:2].copy()
        buildings.geometry = [box(1, 1, 3, 3), box(12, 1, 14, 3)]
        assessor = gpd.GeoDataFrame(
            {
                'AIN': ['2004-001-003', '2004-001-101', '2004-001-102'],
                'assessor_year': [2017, 2017, 2017],
                'SQFTmain': [100, 60, 40],
                'Units': [1, 1, 1],
                'Bedrooms': [2, 1, 1],
                'Bathrooms': [1, 1, 1],
                'YearBuilt': [1950, 1980, 1980],
            },
            geometry=[Point(1, 1), Point(12, 1), Point(13, 1)],
            crs='EPSG:2229',
        )

        building_metrics, site_metrics = build_snapshot_metrics(
            buildings, parcels, assessor, 2017,
            repair_boundary_artifacts=False,
        )

        self.assertEqual(
            building_metrics.attrs['matching_diagnostics']['unmatched_buildings'],
            0,
        )
        null_site = site_metrics.loc[site_metrics['AIN'].isna()].iloc[0]
        self.assertEqual(null_site['site_id'], 'asset:2')
        self.assertEqual(null_site['assessor.ain_count'], 2)
        self.assertEqual(null_site['assessor.sqft'], 100)
        bridge = site_metrics.attrs['site_ain_bridge']
        condo = bridge[bridge['AIN'].isin(['2004001101', '2004001102'])]
        self.assertEqual(set(condo['site_id']), {'asset:2'})
        self.assertEqual(set(condo['assessor.match_method']), {'center_point'})


if __name__ == '__main__':
    unittest.main()
