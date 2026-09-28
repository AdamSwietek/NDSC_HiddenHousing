import unittest

import pandas as pd

from src.identifiers import (
    AINNormalizationError,
    format_ain,
    normalize_ain,
    normalize_ain_series,
)


class NormalizeAINTests(unittest.TestCase):
    def test_normalizes_supported_scalar_forms(self):
        expected = '2004001003'
        self.assertEqual(normalize_ain('2004-001-003'), expected)
        self.assertEqual(normalize_ain(' 2004001003 '), expected)
        self.assertEqual(normalize_ain('2004 001 003'), expected)
        self.assertEqual(normalize_ain(2_004_001_003), expected)
        self.assertEqual(normalize_ain(2_004_001_003.0), expected)
        self.assertEqual(normalize_ain('2004001003.0'), expected)

    def test_preserves_missing_values(self):
        self.assertIsNone(normalize_ain(None))
        self.assertIsNone(normalize_ain(float('nan')))
        self.assertIsNone(normalize_ain(pd.NA))
        self.assertIsNone(normalize_ain(''))

    def test_rejects_malformed_values(self):
        invalid = ['200400103', '20040010030', '2004-ABC-003', '2004001003.5', True]
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(AINNormalizationError):
                    normalize_ain(value)

    def test_coerces_malformed_values(self):
        self.assertIsNone(normalize_ain('not-an-ain', errors='coerce'))

    def test_normalizes_series_and_preserves_index(self):
        source = pd.Series(
            ['2004-001-003', 2_004_001_004.0, None, 'invalid'],
            index=['a', 'b', 'c', 'd'],
            name='AssessorID',
        )
        result = normalize_ain_series(source, errors='coerce', name='AIN')

        self.assertEqual(result.index.tolist(), source.index.tolist())
        self.assertEqual(result.name, 'AIN')
        self.assertEqual(result.iloc[0], '2004001003')
        self.assertEqual(result.iloc[1], '2004001004')
        self.assertTrue(pd.isna(result.iloc[2]))
        self.assertTrue(pd.isna(result.iloc[3]))
        self.assertEqual(str(result.dtype), 'string')

    def test_series_error_reports_bad_index(self):
        source = pd.Series(['2004-001-003', 'bad'], index=[10, 42])
        with self.assertRaisesRegex(AINNormalizationError, r'42'):
            normalize_ain_series(source)

    def test_formats_display_value(self):
        self.assertEqual(format_ain('2004001003'), '2004-001-003')
        self.assertIsNone(format_ain(None))


if __name__ == '__main__':
    unittest.main()
