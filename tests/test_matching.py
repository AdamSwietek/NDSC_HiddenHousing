import unittest

import geopandas as gpd
from shapely.geometry import Polygon, box

from src.matching import aggregate_buildings_to_parcels


class BuildingParcelMatchingTests(unittest.TestCase):
    def setUp(self):
        self.parcels = gpd.GeoDataFrame(
            {'AIN': ['2004-001-003', '2004-001-004', '2004-001-005']},
            geometry=[
                box(0, 0, 10, 10),
                box(10, 0, 20, 10),
                box(30, 0, 40, 10),
            ],
            crs='EPSG:3857',
        )
        self.buildings = gpd.GeoDataFrame(
            {
                'shape.area': [4.0, 12.0, 4.0, 0.0],
                'shape.volume': [40.0, 120.0, 40.0, 0.0],
            },
            geometry=[
                box(1, 1, 3, 3),
                box(8, 1, 14, 3),
                box(50, 0, 52, 2),
                Polygon(),
            ],
            crs='EPSG:3857',
        )

    def test_assigns_once_and_resolves_cross_boundary_by_overlap(self):
        result, aligned = aggregate_buildings_to_parcels(
            self.buildings,
            self.parcels,
            ['shape.area', 'shape.volume'],
            target_crs=3857,
        )

        self.assertEqual(result['shape.n_buildings'].tolist(), [1, 1, 0])
        self.assertEqual(result['shape.area.sum'].tolist(), [4.0, 12.0, 0.0])
        self.assertEqual(result['match.n_overlap_resolved'].tolist(), [0, 1, 0])
        self.assertAlmostEqual(result.loc[1, 'match.overlap_ratio.min'], 2 / 3)
        self.assertTrue(result.geometry.geom_equals(self.parcels.geometry).all())
        self.assertEqual(len(result), len(aligned), 3)

    def test_normalizes_parcel_ains(self):
        result, aligned = aggregate_buildings_to_parcels(
            self.buildings,
            self.parcels,
            ['shape.area'],
            target_crs=3857,
        )
        expected = ['2004001003', '2004001004', '2004001005']
        self.assertEqual(result['AIN'].tolist(), expected)
        self.assertEqual(aligned['AIN'].tolist(), expected)

    def test_reports_unmatched_and_empty_buildings(self):
        result, _ = aggregate_buildings_to_parcels(
            self.buildings,
            self.parcels,
            ['shape.area'],
            target_crs=3857,
        )
        diagnostics = result.attrs['matching_diagnostics']
        self.assertEqual(diagnostics['matched_buildings'], 2)
        self.assertEqual(diagnostics['unmatched_buildings'], 1)
        self.assertEqual(diagnostics['invalid_or_empty_buildings'], 1)

    def test_all_unmatched_buildings_produce_zero_aggregates(self):
        buildings = self.buildings.iloc[[2]].copy()
        result, _ = aggregate_buildings_to_parcels(
            buildings,
            self.parcels,
            ['shape.area'],
            target_crs=3857,
        )
        self.assertEqual(result['shape.area.sum'].tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(result['shape.n_buildings'].tolist(), [0, 0, 0])
        self.assertEqual(
            result.attrs['matching_diagnostics']['unmatched_buildings'], 1
        )

    def test_rejects_non_numeric_aggregate_features(self):
        buildings = self.buildings.assign(label='building')
        with self.assertRaisesRegex(TypeError, 'must be numeric'):
            aggregate_buildings_to_parcels(
                buildings,
                self.parcels,
                ['label'],
                target_crs=3857,
            )


if __name__ == '__main__':
    unittest.main()
