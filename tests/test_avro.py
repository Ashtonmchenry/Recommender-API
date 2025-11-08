from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recommender.avro_utils import load_parsed_schema, assert_valid


def test_reco_response_valid():
    schema = load_parsed_schema(Path("recommender/avro-schemas/RecoResponse.avsc"))
    record = {"user_id": 1, "items": [10, 11], "model_version": "v0.2", "generated_at": 1700000000}
    assert_valid(schema, record)


def test_reco_response_invalid_types():
    schema = load_parsed_schema(Path("recommender/avro-schemas/RecoResponse.avsc"))
    bad = {"user_id": "one", "items": ["x"], "model_version": "v0.2", "generated_at": "now"}
    with pytest.raises(ValueError):
        assert_valid(schema, bad)
