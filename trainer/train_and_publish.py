import json
import os
import shutil
import time
from pathlib import Path

# Where to write versions like v0.3, v0.4, ...
MODEL_REGISTRY = Path(os.getenv("MODEL_REGISTRY", "./model_registry"))

# Provenance fields (can be overridden in env)
DATA_SNAPSHOT_ID = os.getenv("DATA_SNAPSHOT_ID", f"snap-{int(time.time())}")
PIPELINE_GIT_SHA = os.getenv("GIT_SHA", "local-dev")
IMAGE_DIGEST = os.getenv("IMAGE_DIGEST", "unknown")
ARTIFACT_NAME = os.getenv("MODEL_ARTIFACT_NAME", "model.joblib")

MODEL_REGISTRY.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# 1) Work out existing versions and the next version name
# ---------------------------------------------------------------------
existing_dirs = sorted(
    [p for p in MODEL_REGISTRY.iterdir() if p.is_dir() and p.name.startswith("v")]
)

if existing_dirs:
    latest = existing_dirs[-1]
    major, minor = latest.name[1:].split(".")
    new_version = f"v{major}.{int(minor) + 1}"
else:
    latest = None
    new_version = "v1.0"

out_dir = MODEL_REGISTRY / new_version
out_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# 2) Find a source model artifact to copy forward
# ---------------------------------------------------------------------
artifact_source = None
for d in reversed(existing_dirs):
    candidate = d / ARTIFACT_NAME
    if candidate.exists():
        artifact_source = candidate
        break

if artifact_source is not None:
    shutil.copy2(artifact_source, out_dir / ARTIFACT_NAME)
    print(f"Copied model artifact from {artifact_source} -> {out_dir / ARTIFACT_NAME}")
else:
    print("WARNING: no existing model artifact found; no model.joblib written")

# ---------------------------------------------------------------------
# 3) Write provenance metadata (meta.yaml + metadata.json)
# ---------------------------------------------------------------------
created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

meta = {
    "model_version": new_version,
    "created_at": created_at,
    "artifact": ARTIFACT_NAME,
    "data_snapshot_id": DATA_SNAPSHOT_ID,
    "pipeline_git_sha": PIPELINE_GIT_SHA,
    "container_image_digest": IMAGE_DIGEST,
    "metrics": {},  # can be filled with real metrics later
}

# YAML-ish meta file (works with the loader in service/app.py)
meta_yaml_lines = [
    f'model_version: "{meta["model_version"]}"',
    f'created_at: "{meta["created_at"]}"',
    f'artifact: "{meta["artifact"]}"',
    f'data_snapshot_id: "{meta["data_snapshot_id"]}"',
    f'pipeline_git_sha: "{meta["pipeline_git_sha"]}"',
    f'container_image_digest: "{meta["container_image_digest"]}"',
    "metrics: {}",
]
(out_dir / "meta.yaml").write_text("\n".join(meta_yaml_lines) + "\n", encoding="utf-8")

# Optional JSON copy for convenience
(out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

print(f"Published model metadata for {new_version} in {out_dir}")
