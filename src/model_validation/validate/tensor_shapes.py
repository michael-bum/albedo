from __future__ import annotations

import json
import struct
import sys
from functools import lru_cache
from pathlib import Path

_SHOW = 5
_PINNED = Path(__file__).with_name("genesis_tensor_shapes.json")


def read_header(shard: Path) -> dict:
    """One shard's safetensors header (8-byte length prefix, then JSON); no weights are read."""
    with shard.open("rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(n))
    header.pop("__metadata__", None)
    return header


def read_headers(model_dir: str) -> dict[str, dict]:
    """Shard name -> header for every *.safetensors directly under model_dir."""
    return {p.name: read_header(p) for p in sorted(Path(model_dir).glob("*.safetensors"))}


def from_headers(headers: dict[str, dict]) -> dict[str, tuple[int, ...]]:
    """Tensor name -> shape, flattened across every shard of one repo."""
    return {
        name: tuple(info["shape"])
        for header in headers.values()
        for name, info in header.items()
        if name != "__metadata__"
    }


@lru_cache(maxsize=2)
def seed_shapes(seed_dir: str) -> dict[str, tuple[int, ...]]:
    """Cached for the worker's life: make_room does not guard the seed, so it can be evicted
    mid-run and re-reading it would mean re-downloading it."""
    out = from_headers(read_headers(seed_dir))
    if not out:
        raise RuntimeError(f"genesis seed snapshot at {seed_dir} holds no safetensors")
    return out


@lru_cache(maxsize=1)
def pinned_seed_shapes() -> dict[str, tuple[int, ...]]:
    """The seed inventory pinned in-package, so `albedo check-model` can run this check without
    the 72 GB seed. Regenerate from the validator's snapshot when the seed rotates:
    `python -m model_validation.validate.tensor_shapes <seed_dir>`."""
    data = json.loads(_PINNED.read_text())
    return {name: tuple(shape) for name, shape in data["shapes"].items()}


def check(
    candidate: dict[str, tuple[int, ...]], seed: dict[str, tuple[int, ...]]
) -> tuple[bool, str]:
    """Candidate tensor inventory against the genesis seed's.

    Shapes follow from the config.json that the metadata_hash check already pins byte-for-byte, so
    a model that passes that and still differs here was rebuilt, not fine-tuned — training never
    moves a shape.
    """
    missing = sorted(set(seed) - set(candidate))
    extra = sorted(set(candidate) - set(seed))
    wrong = sorted(n for n in set(seed) & set(candidate) if candidate[n] != seed[n])
    if not (missing or extra or wrong):
        return True, ""
    parts = []
    if wrong:
        shown = ", ".join(
            f"{n} is {list(candidate[n])}, expected {list(seed[n])}" for n in wrong[:_SHOW]
        )
        parts.append(f"{len(wrong)} tensor(s) with the wrong shape ({shown})")
    if missing:
        parts.append(f"{len(missing)} missing tensor(s) ({', '.join(missing[:_SHOW])})")
    if extra:
        parts.append(f"{len(extra)} unexpected tensor(s) ({', '.join(extra[:_SHOW])})")
    return False, "model tensors do not match the genesis seed: " + "; ".join(parts)


if __name__ == "__main__":  # regenerate the pinned inventory from a seed snapshot directory
    seed_dir = sys.argv[1]
    seed_ref = sys.argv[2] if len(sys.argv) > 2 else json.loads(_PINNED.read_text())["seed"]
    shapes = {k: list(v) for k, v in sorted(seed_shapes(seed_dir).items())}
    _PINNED.write_text(json.dumps({"seed": seed_ref, "shapes": shapes}, indent=1) + "\n")
    print(f"pinned {len(shapes)} tensor shapes from {seed_dir} into {_PINNED}")
