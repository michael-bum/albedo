from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

SIDECAR = "albedo-king.json"
CURRENT_FILE = "current_king.json"
MODEL_ID_PREFIX = "albedo-king-"


@dataclass(frozen=True)
class KingInfo:
    roman: str
    repo: str
    sha: str
    hotkey: str = ""
    original_repo: str = ""
    downloaded_at: float = 0.0

    @property
    def model_id(self) -> str:
        return f"{MODEL_ID_PREFIX}{self.roman.lower()}" if self.roman else ""

    def public(self) -> dict:
        return {
            "roman": self.roman,
            "model_id": self.model_id,
            "repo": self.repo,
            "sha": self.sha,
            "hotkey": self.hotkey or None,
        }


def read_king_info(model_path: str | Path) -> KingInfo | None:
    return read_king_file(Path(model_path) / SIDECAR)


def read_king_file(file: str | Path) -> KingInfo | None:
    try:
        data = json.loads(Path(file).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not data.get("roman"):
        return None
    return KingInfo(
        roman=str(data["roman"]).upper(),
        repo=str(data.get("repo") or ""),
        sha=str(data.get("sha") or ""),
        hotkey=str(data.get("hotkey") or ""),
        original_repo=str(data.get("original_repo") or ""),
        downloaded_at=float(data.get("downloaded_at") or 0.0),
    )


def write_king_info(model_path: str | Path, info: KingInfo) -> Path:
    return write_king_file(Path(model_path) / SIDECAR, info)


def current_king_file(models_dir: str | Path) -> Path:
    return Path(models_dir) / CURRENT_FILE


def write_king_file(target: str | Path, info: KingInfo) -> Path:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(info), indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, target)
    return target


class KingInfoSource:
    def __init__(self, file: str | Path = "", fixed: KingInfo | None = None) -> None:
        self._path = Path(file) if file else None
        self._fixed = fixed
        self._cached: KingInfo | None = None
        self._mtime: float | None = None

    def get(self) -> KingInfo | None:
        if self._fixed is not None:
            return self._fixed
        if self._path is None:
            return None
        try:
            mtime = self._path.stat().st_mtime
        except OSError:
            self._cached, self._mtime = None, None
            return None
        if mtime != self._mtime:
            self._cached = read_king_file(self._path)
            self._mtime = mtime
        return self._cached
