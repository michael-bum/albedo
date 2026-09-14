from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from model_validation.validate import tensor_shapes

SEED = {
    "model.language_model.embed_tokens.weight": (248320, 2048),
    "model.language_model.layers.0.mlp.experts.gate_up_proj": (256, 1024, 2048),
    "lm_head.weight": (248320, 2048),
}


def test_identical_inventory_passes():
    assert tensor_shapes.check(dict(SEED), SEED) == (True, "")


def test_rank_change_is_rejected():
    """The production case: a candidate whose embed_tokens.weight is not 2-D used to reach dedup
    and die inside canonicalize() as a retryable INFRA_FAULT, five times, costing the miner
    nothing."""
    cand = {**SEED, "model.language_model.embed_tokens.weight": (248320 * 2048,)}
    ok, msg = tensor_shapes.check(cand, SEED)
    assert not ok
    assert "embed_tokens.weight" in msg and "wrong shape" in msg


def test_split_expert_layout_is_rejected_as_missing_and_extra():
    """Fused (E, 2*inter, hidden) is the only layout Qwen3_5MoeExperts declares; a per-expert
    split checkpoint is a different inventory, not an alternative encoding."""
    cand = {k: v for k, v in SEED.items() if "experts" not in k}
    cand["model.language_model.layers.0.mlp.experts.0.gate_proj.weight"] = (512, 2048)
    ok, msg = tensor_shapes.check(cand, SEED)
    assert not ok
    assert "missing" in msg and "unexpected" in msg


def _write_shard(path: Path, tensors: dict[str, tuple[int, ...]]) -> None:
    header = {
        name: {"dtype": "BF16", "shape": list(shape), "data_offsets": [0, 0]}
        for name, shape in tensors.items()
    }
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob)


def test_seed_shapes_reads_every_shard(tmp_path):
    _write_shard(tmp_path / "model-00001-of-00002.safetensors", {"a": (2, 3)})
    _write_shard(tmp_path / "model-00002-of-00002.safetensors", {"b": (4,)})
    assert tensor_shapes.seed_shapes(str(tmp_path)) == {"a": (2, 3), "b": (4,)}


def test_seed_shapes_refuses_an_empty_snapshot(tmp_path):
    """make_room does not guard the seed, so an evicted snapshot must name itself rather than
    silently make every candidate look like it has 'unexpected tensors'."""
    (tmp_path / "gone").mkdir()
    with pytest.raises(RuntimeError, match="no safetensors"):
        tensor_shapes.seed_shapes(str(tmp_path / "gone"))


def test_pinned_seed_inventory_is_this_architecture():
    """The in-package copy the miner CLI checks against must be the validator's seed."""
    pinned = tensor_shapes.pinned_seed_shapes()
    assert len(pinned) == 1045
    assert pinned["model.language_model.embed_tokens.weight"] == (248320, 2048)
    assert pinned["model.language_model.layers.0.mlp.experts.gate_up_proj"] == (256, 1024, 2048)
    assert tensor_shapes.check(pinned, pinned) == (True, "")


def test_read_headers_reads_every_shard_without_the_weights(tmp_path):
    _write_shard(tmp_path / "model-00001-of-00002.safetensors", {"a": (2, 3)})
    _write_shard(tmp_path / "model-00002-of-00002.safetensors", {"b": (4,)})
    (tmp_path / "notes.txt").write_text("ignored")
    headers = tensor_shapes.read_headers(str(tmp_path))
    assert sorted(headers) == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    assert tensor_shapes.from_headers(headers) == {"a": (2, 3), "b": (4,)}


# --- worker wiring --------------------------------------------------------------------------

pytest.importorskip("asyncpg")
pytest.importorskip("torch")

from model_validation import validate_worker as worker  # noqa: E402

BAD_HEADERS = {
    "model.safetensors": {
        "model.language_model.embed_tokens.weight": {"dtype": "BF16", "shape": [248320 * 2048]},
    }
}


def _patch_to_shape_gate(monkeypatch, tmp_path, headers):
    monkeypatch.setattr(worker, "make_ref", lambda repo, digest: object())
    monkeypatch.setattr(worker, "list_files", lambda ref: ["config.json"])
    monkeypatch.setattr(worker, "check_repo", lambda files: (True, ""))
    monkeypatch.setattr(worker, "safetensors_headers", lambda ref: headers)
    monkeypatch.setattr(worker, "check_dtypes", lambda d: (True, ""))
    monkeypatch.setattr(worker, "download_config", lambda ref: str(tmp_path))
    monkeypatch.setattr(worker, "check_chat_template", lambda d, f: (True, ""))
    monkeypatch.setattr(worker, "check_genesis", lambda d, f: (True, ""))
    monkeypatch.setattr(worker.dedup, "ref_dir", lambda: "")
    monkeypatch.setattr(
        worker,
        "seed_shapes",
        lambda seed_dir: {"model.language_model.embed_tokens.weight": (248320, 2048)},
    )
    downloaded = []
    (tmp_path / "model.safetensors").write_bytes(b"x")  # process_model checks the download landed
    monkeypatch.setattr(worker, "make_room", lambda ref, protected_repos=(): None)
    monkeypatch.setattr(
        worker, "download_full", lambda ref: downloaded.append(ref) or str(tmp_path)
    )
    return downloaded


def test_enforced_mismatch_faults_the_miner_without_downloading(monkeypatch, tmp_path):
    downloaded = _patch_to_shape_gate(monkeypatch, tmp_path, BAD_HEADERS)
    monkeypatch.setattr(worker.config, "SHAPE_ENFORCE", True)

    outcome = worker.process_model("repo@sha256:abc", "hotkey")

    assert outcome.fault_class == "MINER_FAULT"
    assert outcome.fault_code == "tensor_shape"
    assert not outcome.retryable  # a strike, not an endless retry loop
    assert downloaded == []  # rejected before the multi-GB download


def test_shadow_mode_never_changes_the_outcome(monkeypatch, tmp_path):
    downloaded = _patch_to_shape_gate(monkeypatch, tmp_path, BAD_HEADERS)
    monkeypatch.setattr(worker.config, "SHAPE_ENFORCE", False)
    monkeypatch.setattr(worker, "check_index", lambda d: (True, ""))
    monkeypatch.setattr(worker.dedup, "run", lambda *a, **k: worker.dedup.GateResult(doc={}))
    monkeypatch.setattr(worker.dedup, "public_summary", lambda res: {"dedup": "pass"})

    outcome = worker.process_model("repo@sha256:abc", "hotkey")

    assert outcome.state == "done"
    assert downloaded  # the pipeline carried on to the download as before


def test_an_unreadable_seed_cannot_fault_a_miner_in_shadow_mode(monkeypatch, tmp_path):
    _patch_to_shape_gate(monkeypatch, tmp_path, BAD_HEADERS)
    monkeypatch.setattr(worker.config, "SHAPE_ENFORCE", False)
    monkeypatch.setattr(
        worker, "seed_shapes", lambda seed_dir: (_ for _ in ()).throw(RuntimeError("gone"))
    )
    monkeypatch.setattr(worker, "check_index", lambda d: (True, ""))
    monkeypatch.setattr(worker.dedup, "run", lambda *a, **k: worker.dedup.GateResult(doc={}))
    monkeypatch.setattr(worker.dedup, "public_summary", lambda res: {"dedup": "pass"})

    assert worker.process_model("repo@sha256:abc", "hotkey").state == "done"
