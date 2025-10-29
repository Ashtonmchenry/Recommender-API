# scripts/check_schemas.py
import os
import sys
import json
from quality.avro_registry import validate_records

TEAM = os.environ.get("TEAM", "aerosparks")
SUBJECT = os.environ.get("SUBJECT", f"{TEAM}.reco_responses-value")

# example: load a small sample batch (replace with your snapshot path or probe data)
SAMPLE_PATH = os.environ.get("SAMPLE_JSON", "sample_records.json")
if not os.path.exists(SAMPLE_PATH):
    # fallback: a trivial shape that should match your schema fields
    sample = [{"request_id":"r1","user_id":1,"movie_id":2,"timestamp":1700000000,"rating":4.0}]
else:
    sample = json.loads(open(SAMPLE_PATH,"r",encoding="utf-8").read())

ok, errs = validate_records(SUBJECT, sample)
if not ok:
    print("Schema validation FAILED:")
    for e in errs[:50]:
        print(" -", e)
    sys.exit(2)

print(f"Schema validation passed for subject={SUBJECT} on {len(sample)} records.")
