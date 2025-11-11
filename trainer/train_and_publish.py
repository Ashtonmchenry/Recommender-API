import json
import os
import time
from pathlib import Path

MODEL_REGISTRY = os.getenv("MODEL_REGISTRY", "./model_registry")
DATA_SNAPSHOT_ID = os.getenv("DATA_SNAPSHOT_ID", f"snap-{int(time.time())}")
PIPELINE_GIT_SHA = os.getenv("GIT_SHA", "local-dev")
IMAGE_DIGEST = os.getenv("IMAGE_DIGEST", "unknown")
reg = Path(MODEL_REGISTRY)
reg.mkdir(parents=True, exist_ok=True)
existing = sorted([p.name for p in reg.iterdir() if p.is_dir() and p.name.startswith("v")])
if existing:
    major, minor = existing[-1][1:].split(".")
    version = f"v{major}.{int(minor) + 1}"
else:
    version = "v1.0"
out = reg / version
out.mkdir(parents=True, exist_ok=True)
meta = {
    "model_version": version,
    "trained_at": int(time.time()),
    "data_snapshot_id": DATA_SNAPSHOT_ID,
    "pipeline_git_sha": PIPELINE_GIT_SHA,
    "container_image_digest": IMAGE_DIGEST,
}
(out / "metadata.json").write_text(json.dumps(meta, indent=2))
print(f"Published model to {out}")
