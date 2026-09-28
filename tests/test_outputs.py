import json
import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from src.outputs import write_feature_outputs
from src.pipeline import FeaturePipelineResult


CRS = 'EPSG:26911'


class FeatureOutputTests(unittest.TestCase):
    def test_writes_complete_validated_run(self):
        result = self._result()
        with tempfile.TemporaryDirectory() as directory:
            output = write_feature_outputs(
                result,
                directory,
                'test_run',
                aoi_wgs84=(-118.4, 33.9, -118.3, 34.0),
                parameters={'snapshot_crs': CRS},
                write_change_geometries=True,
            )

            expected = {
                'building_metrics_2020.parquet',
                'parcel_metrics_2020.parquet',
                'assessor_ain_metrics_2020.parquet',
                'change_metrics_2017_2020.parquet',
                'boundary_repairs_2017.parquet',
                'boundary_repairs_2020.parquet',
                'added_2017_2020.parquet',
                'removed_2017_2020.parquet',
                'manifest.json',
            }
            self.assertEqual({path.name for path in output.iterdir()}, expected)
            buildings = gpd.read_parquet(output / 'building_metrics_2020.parquet')
            self.assertEqual(len(buildings), 1)
            self.assertEqual(buildings.crs.to_epsg(), 26911)
            repairs = pd.read_parquet(output / 'boundary_repairs_2020.parquet')
            self.assertEqual(repairs.loc[0, 'fragment_id'], 'fragment-1')
            self.assertEqual(repairs.loc[0, 'fragment_AIN'], '4223005014')
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(manifest['schema_version'], 4)
            self.assertEqual(manifest['run_name'], 'test_run')
            self.assertEqual(len(manifest['outputs']), 8)

    def test_protects_existing_run_without_overwrite(self):
        result = self._result()
        with tempfile.TemporaryDirectory() as directory:
            write_feature_outputs(
                result, directory, 'test_run', aoi_wgs84=(0, 0, 1, 1)
            )
            with self.assertRaises(FileExistsError):
                write_feature_outputs(
                    result, directory, 'test_run', aoi_wgs84=(0, 0, 1, 1)
                )

    def test_rejects_nested_run_name(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'one non-empty'):
                write_feature_outputs(
                    self._result(),
                    directory,
                    'nested/run',
                    aoi_wgs84=(0, 0, 1, 1),
                )

    def _result(self):
        geometry = box(500000, 3760000, 500010, 3760010)
        buildings = gpd.GeoDataFrame(
            {
                'structure_id': ['building-1'],
                'building_id': ['building-1'],
                'AIN': ['4223005011'],
                'repair.source_building_ids': [
                    ('building-1', 'fragment-1')
                ],
            },
            geometry=[geometry],
            crs=CRS,
        )
        repair_record = {
            'fragment_id': 'fragment-1',
            'dominant_id': 'building-1',
            'fragment_AIN': '4223005014',
            'dominant_AIN': '4223005011',
            'fragment_area_ft2': 60.0,
            'shared_edge_ft': 20.0,
            'parcel_boundary_coverage': 1.0,
            'sliver_buffer_ft': 2.0,
            'opened_retained_fraction': 0.0,
            'fragment_parcel_index': 1,
            'dominant_parcel_index': 0,
        }
        buildings.attrs['boundary_repair_records'] = [repair_record]
        parcels = gpd.GeoDataFrame(
            {
                'site_id': ['ain:4223005011'],
                'AIN': ['4223005011'],
                'building.count': [1],
            },
            geometry=[geometry],
            crs=CRS,
        )
        changes = gpd.GeoDataFrame(
            {
                'reference_site_id': ['ain:4223005011'],
                'reference_AIN': ['4223005011'],
                'change_type': ['no_change'],
            },
            geometry=[geometry],
            crs=CRS,
        )
        added = gpd.GeoDataFrame(
            {'reference_site_id': [], 'reference_AIN': []},
            geometry=gpd.GeoSeries([], crs=CRS), crs=CRS
        )
        removed = added.copy()
        footprint_2017 = gpd.GeoDataFrame(
            {
                'reference_site_id': ['ain:4223005011'],
                'reference_AIN': ['4223005011'],
            },
            geometry=[geometry],
            crs=CRS,
        ).rename_geometry('footprint')
        footprint_2017.attrs['boundary_repair_records'] = []
        footprint_2020 = footprint_2017.copy()
        footprint_2020.attrs['boundary_repair_records'] = [repair_record]
        return FeaturePipelineResult(
            building_metrics={2020: buildings},
            parcel_metrics={2020: parcels},
            assessor_ain_metrics={
                2020: pd.DataFrame({
                    'site_id': ['ain:4223005011'],
                    'AIN': ['4223005011'],
                    'assessor.sqft': [100],
                })
            },
            change_results={
                (2017, 2020): {
                    'metrics': changes,
                    'added': added,
                    'removed': removed,
                    'footprints': {2017: footprint_2017, 2020: footprint_2020},
                }
            },
            diagnostics={'rows': {}},
        )


if __name__ == '__main__':
    unittest.main()
