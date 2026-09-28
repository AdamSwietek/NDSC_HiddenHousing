import unittest

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from src.change_metrics import prepare_epoch_footprints
from src.lariac_cleaning import repair_boundary_fragments
from src.metrics import build_snapshot_metrics


class BoundaryFragmentRepairTests(unittest.TestCase):
    def setUp(self):
        self.parcels = gpd.GeoDataFrame(
            {'AIN': ['2004001003', '2004001004']},
            geometry=[box(0, 0, 10, 10), box(10, 0, 20, 10)],
            crs='EPSG:2229',
        )

    def test_merges_source_fragments_across_a_parcel_boundary(self):
        buildings = self._source_artifact_buildings()

        result = repair_boundary_fragments(buildings, self.parcels)

        self.assertEqual(len(result), 2)
        repaired = result.set_index('building_id').loc[100]
        self.assertAlmostEqual(repaired.geometry.area, 72.4, places=6)
        self.assertEqual(repaired['repair.fragment_count'], 2)
        self.assertAlmostEqual(repaired['repair.absorbed_area_ft2'], 0.4, places=5)
        self.assertEqual(
            repaired['repair.source_building_ids'], ('100', '101', '102')
        )
        self.assertEqual(repaired['repair.method'], 'boundary_fragment_merge')
        diagnostics = result.attrs['boundary_repair_diagnostics']
        self.assertEqual(diagnostics['merged_fragments'], 2)
        self.assertEqual(diagnostics['reconstructed_structures'], 1)

    def test_preserves_a_detached_small_structure(self):
        buildings = self._source_artifact_buildings()

        result = repair_boundary_fragments(buildings, self.parcels)

        self.assertIn(200, set(result['building_id']))
        detached = result.set_index('building_id').loc[200]
        self.assertEqual(detached['repair.fragment_count'], 0)
        self.assertEqual(detached['repair.method'], 'none')

    def test_does_not_merge_similarly_sized_neighbors(self):
        buildings = gpd.GeoDataFrame(
            {
                'building_id': [1, 2],
                'height': [20.0, 20.0],
                'elevation': [100.0, 100.0],
            },
            geometry=[box(1, 1, 10, 9), box(10, 1, 19, 9)],
            crs=self.parcels.crs,
        )

        result = repair_boundary_fragments(buildings, self.parcels)

        self.assertEqual(len(result), 2)
        self.assertEqual(result['repair.fragment_count'].sum(), 0)

    def test_disambiguates_reused_source_building_ids_stably(self):
        buildings = gpd.GeoDataFrame(
            {
                'building_id': [7, 7],
                'height': [8.94, 9.42],
                'elevation': [219.81, 1169.78],
            },
            geometry=[box(1, 1, 5, 5), box(12, 1, 16, 5)],
            crs=self.parcels.crs,
        )

        forward = repair_boundary_fragments(buildings, self.parcels)
        reverse = repair_boundary_fragments(
            buildings.iloc[::-1].reset_index(drop=True), self.parcels
        )

        self.assertTrue(forward['structure_id'].is_unique)
        self.assertTrue(forward['structure_id'].str.startswith('7~').all())
        forward_ids = dict(zip(forward.geometry.map(lambda value: value.wkb), forward['structure_id']))
        reverse_ids = dict(zip(reverse.geometry.map(lambda value: value.wkb), reverse['structure_id']))
        self.assertEqual(forward_ids, reverse_ids)

    def test_cross_parcel_topology_overrides_relative_area(self):
        buildings = gpd.GeoDataFrame(
            {
                'building_id': [1, 2],
                'height': [13.67, 13.63],
                'elevation': [865.55, 865.55],
                'source': ['LARIAC2', 'LARIAC2'],
                'capture_date': [2008, 2008],
            },
            geometry=[
                box(7, 2, 10, 8),      # 18 ft² dominant footprint
                box(10, 2, 10.1, 8),   # 0.6 ft² = 3.33% of dominant
            ],
            crs=self.parcels.crs,
        )

        result = repair_boundary_fragments(buildings, self.parcels)

        self.assertEqual(len(result), 1)
        repaired = result.iloc[0]
        self.assertEqual(repaired['building_id'], 1)
        self.assertEqual(repaired['repair.fragment_count'], 1)
        self.assertAlmostEqual(repaired.geometry.area, 18.6, places=6)

    def test_merges_connected_sliver_larger_than_fifty_square_feet(self):
        parcels = gpd.GeoDataFrame(
            {'AIN': ['4223005011', '4223005014']},
            geometry=[box(0, 0, 20, 30), box(20, 0, 40, 30)],
            crs=self.parcels.crs,
        )
        buildings = gpd.GeoDataFrame(
            {
                'building_id': [1, 2],
                'height': [15.93, 15.92],
                'elevation': [32.12, 32.12],
                'source': ['LARIAC2', 'LARIAC2'],
                'capture_date': [2008, 2008],
            },
            geometry=[
                box(5, 5, 20, 25),   # 300 ft² dominant footprint
                box(20, 5, 23, 25),  # 60 ft², three-foot-wide sliver
            ],
            crs=parcels.crs,
        )

        result = repair_boundary_fragments(buildings, parcels)

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['repair.fragment_count'], 1)
        self.assertAlmostEqual(result.iloc[0].geometry.area, 360.0)
        record = result.attrs['boundary_repair_records'][0]
        self.assertEqual(record['opened_retained_fraction'], 0.0)

    def test_preserves_connected_compact_footprint(self):
        parcels = gpd.GeoDataFrame(
            {'AIN': ['4223005011', '4223005014']},
            geometry=[box(0, 0, 20, 30), box(20, 0, 40, 30)],
            crs=self.parcels.crs,
        )
        buildings = gpd.GeoDataFrame(
            {
                'building_id': [1, 2],
                'height': [15.0, 15.0],
                'elevation': [32.0, 32.0],
            },
            geometry=[
                box(5, 5, 20, 25),
                box(20, 10, 30, 20),  # compact 100 ft² footprint
            ],
            crs=parcels.crs,
        )

        result = repair_boundary_fragments(buildings, parcels)

        self.assertEqual(len(result), 2)
        self.assertEqual(result['repair.fragment_count'].sum(), 0)

    def test_snapshot_pipeline_repairs_before_assignment_and_aggregation(self):
        buildings = self._source_artifact_buildings().assign(lariac_year=2017)
        assessor = pd.DataFrame(
            {
                'AIN': self.parcels['AIN'],
                'assessor_year': [2017, 2017],
                'SQFTmain': [72, 1],
            }
        )

        building_metrics, parcel_metrics = build_snapshot_metrics(
            buildings, self.parcels, assessor, 2017
        )
        parcels = parcel_metrics.set_index('AIN')

        self.assertEqual(len(building_metrics), 2)
        self.assertEqual(parcels.loc['2004001003', 'building.count'], 1)
        self.assertAlmostEqual(
            parcels.loc['2004001003', 'building.footprint_area'], 72.4
        )
        self.assertEqual(parcels.loc['2004001004', 'building.count'], 1)
        self.assertAlmostEqual(
            parcels.loc['2004001004', 'building.footprint_area'], 0.2
        )
        self.assertEqual(
            parcel_metrics.attrs['boundary_repair_diagnostics'][
                'merged_fragments'
            ],
            2,
        )

    def test_change_pipeline_repairs_in_metric_crs(self):
        parcels = gpd.GeoDataFrame(
            {'AIN': ['2004001003', '2004001004']},
            geometry=[box(0, 0, 3.048, 3.048), box(3.048, 0, 6.096, 3.048)],
            crs='EPSG:26911',
        )
        buildings = gpd.GeoDataFrame(
            {
                'building_id': [100, 101],
                'height': [20.0, 20.2],
                'elevation': [100.0, 100.0],
            },
            geometry=[
                box(0.3, 0.3, 3.048, 2.7),
                box(3.048, 0.6, 3.07848, 1.6),
            ],
            crs=parcels.crs,
        )

        result = prepare_epoch_footprints(buildings, parcels, 2017)

        self.assertEqual(len(result), 1)
        self.assertEqual(result.loc[0, 'reference_AIN'], '2004001003')
        self.assertEqual(result.loc[0, 'building_count'], 1)
        self.assertEqual(
            result.attrs['boundary_repair_diagnostics']['merged_fragments'], 1
        )

    def _source_artifact_buildings(self):
        return gpd.GeoDataFrame(
            {
                'building_id': [100, 101, 102, 200],
                'height': [20.0, 20.2, 19.9, 10.0],
                'elevation': [100.0, 100.0, 100.0, 110.0],
                'source': ['LARIAC2', 'LARIAC2', 'LARIAC2', 'LARIAC2'],
                'capture_date': [2008, 2008, 2008, 2008],
            },
            geometry=[
                box(1, 1, 10, 9),
                box(10, 2, 10.1, 4),
                box(10, 6, 10.1, 8),
                box(12, 1, 12.1, 3),
            ],
            crs=self.parcels.crs,
        )


if __name__ == '__main__':
    unittest.main()
