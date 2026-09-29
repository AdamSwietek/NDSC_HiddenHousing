import unittest

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.viz import plot_ain_hexbin


def _points():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        'AIN': [f'{i:010d}' for i in range(1, 201)],
        'x': rng.uniform(0, 10_000, 200),
        'y': rng.uniform(0, 5_000, 200),
    })


class PlotAINHexbinTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_maps_numeric_metric_and_reports_missing_points(self):
        points = _points()
        data = pd.DataFrame({
            'AIN': [*points['AIN'], '9999999999'],
            'score': np.linspace(0, 1, 201),
        })
        ax = plot_ain_hexbin(
            data, 'score', points=points, hex_width_ft=2_500, min_count=1
        )
        hexes = ax.collections[0]
        self.assertGreater(len(hexes.get_array()), 0)
        self.assertTrue(np.all((hexes.get_array() >= 0) & (hexes.get_array() <= 1)))
        self.assertIn('200 AINs mapped; 1 without a center point', ax.texts[0].get_text())

    def test_accepts_boolean_metric_and_dashed_ains(self):
        points = _points()
        data = pd.DataFrame({
            'AIN': [f'{a[:4]}-{a[4:7]}-{a[7:]}' for a in points['AIN']],
            'flag': [True, False] * 100,
        })
        ax = plot_ain_hexbin(
            data, 'flag', points=points, reduce_function=np.mean, min_count=1
        )
        self.assertIn('200 AINs mapped; 0 without', ax.texts[0].get_text())

    def test_rejects_missing_metric_column(self):
        with self.assertRaises(ValueError):
            plot_ain_hexbin(pd.DataFrame({'AIN': []}), 'score', points=_points())


if __name__ == '__main__':
    unittest.main()
