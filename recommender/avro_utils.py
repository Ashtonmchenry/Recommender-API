# recommender/avro_utils.py
import json
from pathlib import Path

from fastavro import parse_schema
from fastavro.validation import validate

try:
    # fastavro ≥1.9 exposes this
    from fastavro._validate_common import ValidationError as FastAvroValidationError  # type: ignore
except Exception:  # fallback if symbol moves
    FastAvroValidationError = Exception  # broad catch

_CACHE = {}


def load_schema(path: Path) -> dict:
    key = str(path.resolve())
    if key not in _CACHE:
        _CACHE[key] = parse_schema(json.loads(path.read_text()))
    return _CACHE[key]


# optional alias if test uses load_parsed_schema
def load_parsed_schema(path: Path) -> dict:
    return load_schema(path)


def assert_valid(schema: dict, record: dict) -> None:
    try:
        ok = validate(record, schema)  # some versions return bool, some may raise
    except FastAvroValidationError as e:
        raise ValueError(f"Avro validation failed: {e}") from None
    except Exception as e:
        # guard against any other internal error shapes
        raise ValueError(f"Avro validation failed: {e}") from None

    if not ok:
        # older versions return False instead of raising
        raise ValueError(f"Avro validation failed for {schema.get('name')}: {record}")
