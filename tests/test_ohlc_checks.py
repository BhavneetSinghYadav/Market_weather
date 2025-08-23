from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from mw.utils.ohlc_checks import validate_ohlc


def test_validate_ohlc_various_rows():
    df = pd.DataFrame(
        {
            "open": [1, 2, 1.5, 1, 1],
            "high": [2.5, 2.5, 1.8, 1, 0.5],
            "low": [0.5, 1.5, 1.0, 1, 0.5],
            "close": [2, 1, 2, 1, 1],
        }
    )
    # rows: valid, low>min, max>high, flat valid, flat invalid
    expected = pd.Series([True, False, False, True, False], dtype=bool)
    result = validate_ohlc(df)
    pd.testing.assert_series_equal(result, expected, check_names=False)

