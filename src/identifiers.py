"""Canonical identifiers shared across assessor, parcel, and LARIAC data."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from typing import Optional

import pandas as pd


AIN_DIGITS = 10
_MISSING_TEXT = {'', 'nan', 'none', 'null', '<na>'}
_INTEGER_DECIMAL = re.compile(r'^(\d+)\.0+$')
_AIN_FORMATTING = re.compile(r'[\s-]+')


class AINNormalizationError(ValueError):
    """Raised when a non-missing value cannot be normalized as an LA County AIN."""


def normalize_ain(value, *, errors: str = 'raise') -> Optional[str]:
    """Return an LA County AIN as a compact 10-digit string.

    Accepted representations include ``"2004-001-003"``, ``"2004001003"``,
    integer values, and integer-valued floats. Missing values return ``None`` so
    callers can use spatial matching as a fallback.

    Parameters
    ----------
    value:
        Scalar parcel identifier.
    errors:
        ``"raise"`` rejects malformed non-missing values. ``"coerce"`` returns
        ``None`` for them.
    """
    _validate_errors(errors)
    if _is_missing(value):
        return None

    if isinstance(value, bool):
        return _invalid_ain(value, errors)
    if isinstance(value, int):
        candidate = str(value)
    elif isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            return _invalid_ain(value, errors)
        candidate = str(int(value))
    else:
        candidate = str(value).strip()
        if candidate.lower() in _MISSING_TEXT:
            return None
        decimal_match = _INTEGER_DECIMAL.fullmatch(candidate)
        if decimal_match:
            candidate = decimal_match.group(1)

    candidate = _AIN_FORMATTING.sub('', candidate)
    if len(candidate) != AIN_DIGITS or not candidate.isdigit():
        return _invalid_ain(value, errors)
    return candidate


def normalize_ain_series(
    values: Iterable,
    *,
    errors: str = 'raise',
    name: Optional[str] = None,
) -> pd.Series:
    """Vectorized AIN normalization that preserves index and missing values."""
    _validate_errors(errors)
    source = values.copy() if isinstance(values, pd.Series) else pd.Series(values)
    text = source.astype('string').str.strip()
    missing = text.isna() | text.str.lower().isin(_MISSING_TEXT)
    normalized = (
        text
        .str.replace(r'^(\d+)\.0+$', r'\1', regex=True)
        .str.replace(r'[\s-]+', '', regex=True)
    )
    valid = normalized.str.fullmatch(rf'\d{{{AIN_DIGITS}}}', na=False)
    invalid = ~missing & ~valid

    if invalid.any() and errors == 'raise':
        examples = [
            f'{index!r}: {source.loc[index]!r}'
            for index in source.index[invalid][:5]
        ]
        raise AINNormalizationError(
            'Invalid AIN values at index/value ' + ', '.join(examples)
        )

    result = normalized.where(valid, pd.NA).astype('string')
    result.name = source.name if name is None else name
    return result


def format_ain(value, *, errors: str = 'raise') -> Optional[str]:
    """Return the display form ``XXXX-XXX-XXX`` for a scalar AIN."""
    normalized = normalize_ain(value, errors=errors)
    if normalized is None:
        return None
    return f'{normalized[:4]}-{normalized[4:7]}-{normalized[7:]}'


def _is_missing(value) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if isinstance(missing, bool) else False


def _validate_errors(errors: str) -> None:
    if errors not in {'raise', 'coerce'}:
        raise ValueError("errors must be 'raise' or 'coerce'")


def _invalid_ain(value, errors: str) -> Optional[str]:
    if errors == 'coerce':
        return None
    raise AINNormalizationError(
        f'AIN must contain exactly {AIN_DIGITS} digits; received {value!r}'
    )
