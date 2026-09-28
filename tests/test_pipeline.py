import unittest
from unittest.mock import patch

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from src.pipeline import ChangeJob, SnapshotJob, run_feature_pipeline


CRS = 'EPSG:26911'


class FeaturePipelineTests(unittest.TestCase):
    def setUp(self):
        self.parcels = {
            2017: self._parcels(2017),
            2021: self._parcels(2021),
        }
        self.buildings = {
            2017: self._buildings(2017, 500001),
            2020: self._buildings(2020, 500002),
        }
        self.assessor = pd.DataFrame(
            {
                'AIN': ['4223005014'],
                'assessor_year': [2020],
                'SQFTmain': [100],
                'Units': [1],
                'Bedrooms': [2],
                'Bathrooms': [1],
                'YearBuilt': [1950],
            }
        )

    @patch('src.pipeline.load_assessor_columns')
    @patch('src.pipeline.load_parcel_polygons')
    @patch('src.pipeline.load_lariac')
    def test_loads_each_source_once_and_runs_both_products(
        self, load_lariac, load_parcels, load_assessor
    ):
        load_lariac.side_effect = lambda year, *args, **kwargs: self.buildings[year]
        load_parcels.side_effect = lambda year, *args, **kwargs: self.parcels[year]
        load_assessor.return_value = self.assessor

        result = run_feature_pipeline(
            (-118.44, 33.98, -118.43, 33.99),
            snapshot_jobs=[SnapshotJob(2020, 2021)],
            change_jobs=[ChangeJob(2017, 2020, 2017)],
            snapshot_crs=CRS,
            change_crs=CRS,
        )

        self.assertEqual(load_lariac.call_count, 2)
        self.assertEqual(
            {call.args[0] for call in load_lariac.call_args_list},
            {2017, 2020},
        )
        self.assertEqual(load_parcels.call_count, 2)
        self.assertEqual(
            {call.args[0] for call in load_parcels.call_args_list},
            {2017, 2021},
        )
        self.assertEqual(load_assessor.call_count, 2)
        self.assertEqual(set(result.building_metrics), {2020})
        self.assertEqual(set(result.parcel_metrics), {2020})
        self.assertEqual(set(result.change_results), {(2017, 2020)})
        self.assertEqual(result.row_summary(), {
            'building_metrics_2020': 1,
            'parcel_metrics_2020': 1,
            'assessor_ain_metrics_2020': 1,
            'change_metrics_2017_2020': 1,
        })

    def test_rejects_duplicate_snapshot_years(self):
        with self.assertRaisesRegex(ValueError, 'only once'):
            run_feature_pipeline(
                (0, 0, 1, 1),
                snapshot_jobs=[SnapshotJob(2020, 2021), SnapshotJob(2020, 2022)],
            )

    def test_requires_at_least_one_job(self):
        with self.assertRaisesRegex(ValueError, 'at least one'):
            run_feature_pipeline((0, 0, 1, 1))

    def _parcels(self, year):
        frame = gpd.GeoDataFrame(
            {'AIN': ['4223005014'], 'parcel_year': [year]},
            geometry=[box(500000, 3760000, 500020, 3760020)],
            crs=CRS,
        )
        frame.attrs['loading_diagnostics'] = {
            'source_rows': 1,
            'missing_ain_rows': 0,
            'returned_rows': 1,
            'included_missing_ain': False,
        }
        return frame

    def _buildings(self, year, building_id):
        return gpd.GeoDataFrame(
            {
                'building_id': [building_id],
                'height': [10.0],
                'elevation': [100.0],
                'status': ['Unchanged'],
                'source': [f'LARIAC{year}'],
                'capture_date': [year],
                'old_building_id': [None],
                'lariac_year': [year],
            },
            geometry=[box(500002, 3760002, 500012, 3760012)],
            crs=CRS,
        )


if __name__ == '__main__':
    unittest.main()
