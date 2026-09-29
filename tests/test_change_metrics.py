import unittest

import geopandas as gpd
from shapely import affinity
from shapely.geometry import Polygon, box

from src.change_metrics import (
    _difference_series,
    calculate_change_metrics,
    estimate_epoch_shift,
    prepare_epoch_footprints,
    remove_slivers,
)


CRS = 'EPSG:26911'


def prepared(year, items):
    rows = []
    geometries = []
    for ain, geometry, count in items:
        rows.append(
            {
                'reference_site_id': f'ain:{ain}',
                'reference_AIN': ain,
                'year': year,
                'footprint_area_m2': geometry.area,
                'building_count': count,
            }
        )
        geometries.append(geometry)
    frame = gpd.GeoDataFrame(
        rows,
        geometry=gpd.GeoSeries(geometries, crs=CRS),
        crs=CRS,
    )
    return frame.rename_geometry('footprint')


class SliverRemovalTests(unittest.TestCase):
    def test_removes_thin_strip(self):
        thin_strip = box(0, 0, 10, 0.5)
        self.assertTrue(remove_slivers(thin_strip, 0.4).is_empty)

    def test_preserves_substantial_rectangle_and_sharp_corners(self):
        building = box(0, 0, 5, 5)
        cleaned = remove_slivers(building, 0.4)
        self.assertAlmostEqual(cleaned.area, building.area, places=6)
        self.assertTrue(cleaned.equals(building))
        self.assertEqual(len(cleaned.exterior.coords), 5)

    def test_difference_repairs_topology_error(self):
        invalid_bowtie = Polygon([
            (0, 0), (2, 2), (0, 2), (2, 0), (0, 0),
        ])
        left = gpd.GeoSeries([invalid_bowtie], crs=CRS)
        right = gpd.GeoSeries([box(0, 0, 1, 2)], crs=CRS)

        result = _difference_series(left, right)

        self.assertTrue(result.iloc[0].is_valid)
        self.assertAlmostEqual(result.iloc[0].area, 1.0)

    def test_difference_extracts_polygon_from_mixed_dimension_repair(self):
        polygon_with_line_spike = Polygon([
            (0, 0), (2, 0), (2, 2), (1, 2),
            (1, 3), (1, 2), (0, 2), (0, 0),
        ])
        left = gpd.GeoSeries([polygon_with_line_spike], crs=CRS)
        right = gpd.GeoSeries([box(0, 0, 1, 3)], crs=CRS)

        result = _difference_series(left, right)

        self.assertTrue(result.iloc[0].is_valid)
        self.assertIn(result.iloc[0].geom_type, ('Polygon', 'MultiPolygon'))
        self.assertAlmostEqual(result.iloc[0].area, 2.0)


class ChangeMetricsTests(unittest.TestCase):
    def setUp(self):
        self.ains = [f'20040010{number:02d}' for number in range(1, 7)]
        self.reference = gpd.GeoDataFrame(
            {'AIN': self.ains},
            geometry=[box(i * 20, -5, i * 20 + 15, 15) for i in range(6)],
            crs=CRS,
        )

    def test_prepares_dissolved_reference_ain_footprints(self):
        buildings = gpd.GeoDataFrame(
            {'building_id': [1, 2, 3]},
            geometry=[
                box(1, 1, 3, 3),
                box(4, 1, 6, 3),
                box(200, 1, 202, 3),
            ],
            crs=CRS,
        )
        result = prepare_epoch_footprints(
            buildings, self.reference.iloc[:2], 2017
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result.loc[0, 'reference_AIN'], self.ains[0])
        self.assertEqual(
            result.loc[0, 'reference_site_id'], f'ain:{self.ains[0]}'
        )
        self.assertEqual(result.loc[0, 'building_count'], 2)
        self.assertEqual(result.loc[0, 'footprint_area_m2'], 8.0)
        diagnostics = result.attrs['assignment_diagnostics']
        self.assertEqual(diagnostics['matched_buildings'], 2)
        self.assertEqual(diagnostics['unmatched_buildings'], 1)

    def test_includes_null_ain_reference_site(self):
        reference = gpd.GeoDataFrame(
            {
                'AIN': [self.ains[0], None],
                'ASSETID': [1, 99],
            },
            geometry=[box(0, 0, 10, 10), box(10, 0, 20, 10)],
            crs=CRS,
        )
        buildings = gpd.GeoDataFrame(
            {'building_id': [1, 2]},
            geometry=[box(1, 1, 3, 3), box(12, 1, 14, 3)],
            crs=CRS,
        )

        result = prepare_epoch_footprints(
            buildings, reference, 2017, repair_boundary_artifacts=False
        )

        self.assertEqual(len(result), 2)
        self.assertEqual(
            result.attrs['assignment_diagnostics']['unmatched_buildings'], 0
        )
        null_site = result.loc[result['reference_AIN'].isna()].iloc[0]
        self.assertEqual(null_site['reference_site_id'], 'asset:99')
        self.assertEqual(null_site['building_count'], 1)

    def test_calculates_cleaned_differences_and_change_classes(self):
        a, b, c, d, e, f = self.ains
        start = prepared(
            2017,
            [
                (a, box(0, 0, 4, 4), 1),
                (b, box(20, 0, 24, 4), 1),
                (d, box(60, 0, 64, 4), 1),
                (e, box(80, 0, 84, 4), 1),
                (f, box(100, 0, 110, 10), 1),
            ],
        )
        end = prepared(
            2020,
            [
                (a, box(0, 0, 4, 4), 1),
                (b, box(20, 0, 26, 4), 1),
                (c, box(40, 0, 44, 4), 1),
                (e, box(85, 0, 89, 4), 1),
                (f, box(100.2, 0, 110.2, 10), 1),
            ],
        )

        result = calculate_change_metrics(
            start,
            end,
            self.reference,
            align=False,
            min_change_area_m2=1.0,
            relative_change_threshold=0.01,
            store_geometries=True,
        )
        metrics = result['metrics'].set_index('reference_AIN')

        self.assertEqual(metrics.loc[a, 'change_type'], 'no_change')
        self.assertEqual(metrics.loc[b, 'change_type'], 'extended')
        self.assertEqual(metrics.loc[c, 'change_type'], 'new_development')
        self.assertEqual(metrics.loc[d, 'change_type'], 'demolished')
        self.assertEqual(
            metrics.loc[e, 'change_type'], 'replacement_or_reconfiguration'
        )
        self.assertEqual(metrics.loc[f, 'change_type'], 'possible_misalignment')
        self.assertAlmostEqual(metrics.loc[b, 'area_added_m2'], 8.0, places=3)
        self.assertAlmostEqual(metrics.loc[b, 'area_net_m2'], 8.0, places=3)
        self.assertGreater(metrics.loc[f, 'area_added_raw_m2'], 0)
        self.assertEqual(metrics.loc[f, 'area_added_m2'], 0)
        self.assertGreater(metrics.loc[f, 'area_filtered_added_m2'], 0)
        self.assertIn(f, set(result['filtered_added']['reference_AIN']))
        self.assertIn(f, set(result['raw_added']['reference_AIN']))
        self.assertNotIn(f, set(result['added']['reference_AIN']))

    def test_estimates_and_applies_robust_epoch_shift(self):
        items = [
            (ain, box(i * 20, 0, i * 20 + 5, 5), 1)
            for i, ain in enumerate(self.ains)
        ]
        start = prepared(2017, items)
        end = prepared(
            2020,
            [
                (ain, affinity.translate(geometry, xoff=2, yoff=-1), count)
                for ain, geometry, count in items
            ],
        )
        alignment = estimate_epoch_shift(start, end)

        self.assertAlmostEqual(alignment['shift_dx_m'], 2.0)
        self.assertAlmostEqual(alignment['shift_dy_m'], -1.0)
        result = calculate_change_metrics(start, end, self.reference)
        self.assertTrue(result['metrics']['change_type'].eq('no_change').all())
        self.assertTrue(result['diagnostics']['alignment_applied'])

    def test_applies_precomputed_alignment_for_tiled_runs(self):
        items = [
            (ain, box(i * 20, 0, i * 20 + 5, 5), 1)
            for i, ain in enumerate(self.ains[:2])
        ]
        start = prepared(2017, items)
        end = prepared(2020, [
            (ain, affinity.translate(geometry, xoff=2, yoff=-1), count)
            for ain, geometry, count in items
        ])

        result = calculate_change_metrics(
            start,
            end,
            self.reference,
            alignment_override={'shift_dx_m': 2, 'shift_dy_m': -1},
        )

        self.assertTrue(result['metrics']['change_type'].eq('no_change').all())
        self.assertEqual(result['diagnostics']['shift_dx_m'], 2.0)
        self.assertEqual(result['diagnostics']['shift_dy_m'], -1.0)

    def test_handles_an_epoch_with_no_buildings(self):
        empty_buildings = gpd.GeoDataFrame(
            {'building_id': []}, geometry=gpd.GeoSeries([], crs=CRS), crs=CRS
        )
        end_buildings = gpd.GeoDataFrame(
            {'building_id': [1]}, geometry=[box(1, 1, 3, 3)], crs=CRS
        )
        start = prepare_epoch_footprints(
            empty_buildings, self.reference.iloc[:1], 2017
        )
        end = prepare_epoch_footprints(
            end_buildings, self.reference.iloc[:1], 2020
        )

        result = calculate_change_metrics(
            start, end, self.reference.iloc[:1], align=False
        )
        self.assertEqual(
            result['metrics'].loc[0, 'change_type'], 'new_development'
        )


if __name__ == '__main__':
    unittest.main()
