import unittest

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from src.city_pipeline import (
    _disambiguate_reused_structure_ids,
    _normalize_output_identifier_types,
    _require_full_lariac_coverage,
    make_processing_tiles,
)
from src.pipeline import ChangeJob, SnapshotJob


class CityPipelineTests(unittest.TestCase):
    def test_rejects_partial_2023_source_before_city_run(self):
        with self.assertRaisesRegex(RuntimeError, '2023.*partial_fire_areas'):
            _require_full_lariac_coverage(
                [SnapshotJob(2017, 2017)],
                [ChangeJob(2020, 2023, 2017)],
            )

    def test_tiles_cover_aoi_once_and_add_halos(self):
        aoi = gpd.GeoDataFrame(
            {'name': ['test']},
            geometry=[box(500000, 3760000, 516000, 3776000)],
            crs='EPSG:26911',
        )

        tiles, grid = make_processing_tiles(
            aoi,
            tile_size_miles=5,
            halo_feet=500,
            working_crs='EPSG:26911',
        )

        self.assertEqual(len(tiles), 4)
        self.assertAlmostEqual(
            sum(tile.core.area for tile in tiles), aoi.geometry.iloc[0].area
        )
        self.assertTrue(all(tile.halo.area > tile.core.area for tile in tiles))
        self.assertEqual(grid.max_column, 1)
        self.assertEqual(grid.max_row, 1)

    def test_normalizes_building_ids_from_older_numeric_checkpoints(self):
        older = pd.DataFrame({'structure_id': [2014102531960000]})
        newer = pd.DataFrame({'structure_id': ['2014102531960000~abc123']})

        normalized = [
            _normalize_output_identifier_types(frame, 'building_metrics_2017.parquet')
            for frame in (older, newer)
        ]
        merged = pd.concat(normalized, ignore_index=True)

        self.assertEqual(
            merged['structure_id'].tolist(),
            ['2014102531960000', '2014102531960000~abc123'],
        )

    def test_disambiguates_source_id_reused_across_tiles(self):
        merged = gpd.GeoDataFrame(
            {'structure_id': ['7', '7']},
            geometry=[box(0, 0, 1, 1), box(10, 10, 11, 11)],
            crs='EPSG:2229',
        )

        result = _disambiguate_reused_structure_ids(merged)

        self.assertTrue(result['structure_id'].is_unique)
        self.assertTrue(result['structure_id'].str.startswith('7~').all())


if __name__ == '__main__':
    unittest.main()
