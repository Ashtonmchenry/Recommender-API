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

# Find the latest existing version folder (v0.1, v0.2, ...)
existing = sorted(
    [p.name for p in MODEL_REGISTRY.iterdir() if p.is_dir() and p.name.startswith("v")]
)

if existing:
    major, minor = existing[-1][1:].split(".")
    version = f"v{major}.{int(minor) + 1}"
    prev_dir = MODEL_REGISTRY / existing[-1]
else:
    version = "v1.0"
    prev_dir = None

out = MODEL_REGISTRY / version
out.mkdir(parents=True, exist_ok=True)

# 1) Copy the previous model artifact forward if we have one
if prev_dir is not None:
    prev_artifact = prev_dir / ARTIFACT_NAME
    if prev_artifact.exists():
        shutil.copy2(prev_artifact, out / ARTIFACT_NAME)

# 2) Build metadata / provenance
created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

meta = {
    "model_version": version,
    "created_at": created_at,
    "artifact": ARTIFACT_NAME,
    "data_snapshot_id": DATA_SNAPSHOT_ID,
    "pipeline_git_sha": PIPELINE_GIT_SHA,
    "container_image_digest": IMAGE_DIGEST,
    "metrics": {},  # can fill in later
}

# meta.yaml (what the service expects by default)
meta_yaml_lines = [
    f'model_version: "{meta["model_version"]}"',
    f'created_at: "{meta["created_at"]}"',
    f'artifact: "{meta["artifact"]}"',
    f'data_snapshot_id: "{meta["data_snapshot_id"]}"',
    f'pipeline_git_sha: "{meta["pipeline_git_sha"]}"',
    f'container_image_digest: "{meta["container_image_digest"]}"',
    "metrics: {}",
]
(out / "meta.yaml").write_text("\n".join(meta_yaml_lines) + "\n", encoding="utf-8")

# Optional JSON copy (convenient for scripts)
(out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

print(f"Published model metadata for {version} in {out}")
