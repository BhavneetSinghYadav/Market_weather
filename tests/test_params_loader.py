import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from mw.utils.params import Params, load_params  # noqa: E402


def test_default_params_match_json():
    params = Params()
    loaded = load_params()
    assert loaded == params
