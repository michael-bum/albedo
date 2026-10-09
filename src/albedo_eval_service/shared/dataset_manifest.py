from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

DEFAULT_DATASET_MANIFEST_HASH = "2662d2be0d34b1f9aebd25632a7d006d4d7fc45acadee170542dbcf96fba0c31"


def load_manifest_file(path: str | Path, *, expected_sha256: str) -> dict[str, Any]:

    manifest_path = Path(path)
    payload = manifest_path.read_bytes()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    normalized_expected = expected_sha256.removeprefix("sha256:")
    if normalized_expected and actual_sha256 != normalized_expected:
        raise ValueError(
            f"dataset manifest hash mismatch: expected {normalized_expected}, got {actual_sha256}"
        )
    loaded = json.loads(payload)
    if not isinstance(loaded, dict):
        raise ValueError("dataset manifest must be a JSON object")
    return loaded
