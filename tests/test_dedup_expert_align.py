from __future__ import annotations

import base64

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("opensearchpy")

from test_dedup_canon import _write_model  # noqa: E402

from model_validation.dedup import bank, gate, sketch  # noqa: E402
from model_validation.dedup.canon import ExpertAlign  # noqa: E402
from model_validation.dedup.signals import mats, rel_dist  # noqa: E402
from model_validation.dedup.verdict import Verdict  # noqa: E402

SECRET = b"unit-test-secret-0123456789abcdef-0123456789"
CPU = torch.device("cpu")
ALIGN = ExpertAlign(max_router_ident=0.5, sure=0.9, floor=0.6)
RES, INTER, EXPERTS, VOCAB = 32, 8, 8, 64
P = "model.language_model.layers.0.mlp."
DERANGE = torch.tensor([(i + 3) % EXPERTS for i in range(EXPERTS)])


def _base(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "model.language_model.embed_tokens.weight": torch.randn(VOCAB, RES, generator=g),
        "lm_head.weight": torch.randn(VOCAB, RES, generator=g),
        "model.language_model.layers.0.self_attn.q_proj.weight": torch.randn(24, RES, generator=g),
        "model.language_model.layers.0.self_attn.o_proj.weight": torch.randn(RES, 24, generator=g),
        P + "gate.weight": torch.randn(EXPERTS, RES, generator=g),
        P + "experts.gate_up_proj": torch.randn(EXPERTS, 2 * INTER, RES, generator=g),
        P + "experts.down_proj": torch.randn(EXPERTS, RES, INTER, generator=g),
    }


def _shuffled(t, perm, parts=("router", "experts")):
    out = dict(t)
    if "router" in parts:
        out[P + "gate.weight"] = t[P + "gate.weight"][perm]
    if "experts" in parts:
        out[P + "experts.gate_up_proj"] = t[P + "experts.gate_up_proj"][perm]
        out[P + "experts.down_proj"] = t[P + "experts.down_proj"][perm]
    return out


def _fp(tmp_path, name, tensors, ref, align=ALIGN):
    d = tmp_path / name
    d.mkdir()
    _write_model(d, {k: v.to(torch.bfloat16) for k, v in tensors.items()}, 1)
    return sketch.fingerprint(str(d), str(ref), SECRET, CPU, model_uri=name, align=align)


@pytest.fixture
def ref(tmp_path):
    d = tmp_path / "ref"
    d.mkdir()
    _write_model(d, {k: v.to(torch.bfloat16) for k, v in _base().items()}, 1)
    return d


def _experts(doc):
    return {t["name"]: (t["s"], t["x"]) for t in doc["tensors"] if "experts." in t["name"]}


def _worst_expert_match(doc):
    return min(t["cos"] for t in doc["tensors"] if "experts." in t["name"])


def test_shuffled_experts_and_router_are_put_back(tmp_path, ref):
    base = _fp(tmp_path, "base", _base(), ref)
    moved = _fp(tmp_path, "moved", _shuffled(_base(), DERANGE), ref)
    assert moved["realigned_layers"] == ["layers.0.mlp."]
    assert _experts(moved) == _experts(base)
    assert rel_dist(mats(moved), mats(base)) < 1e-6


def test_live_path_still_sees_a_shuffled_copy_as_misaligned(tmp_path, ref):
    moved = _fp(tmp_path, "moved", _shuffled(_base(), DERANGE), ref, align=None)
    assert "realigned_layers" not in moved
    assert _worst_expert_match(moved) < ALIGN.floor


def test_standard_order_fingerprint_is_unchanged_by_alignment(tmp_path, ref):
    g = torch.Generator().manual_seed(7)
    trained = _base()
    trained[P + "experts.down_proj"] = trained[P + "experts.down_proj"] + 0.05 * torch.randn(
        EXPERTS, RES, INTER, generator=g
    )
    with_align = _fp(tmp_path, "a", trained, ref)
    without = _fp(tmp_path, "b", trained, ref, align=None)
    with_align.pop("secs"), without.pop("secs")
    with_align.pop("model_uri"), without.pop("model_uri")
    assert with_align == without


def test_router_only_shuffle_keeps_the_experts_where_they_are(tmp_path, ref):
    base = _fp(tmp_path, "base", _base(), ref)
    broken = _fp(tmp_path, "broken", _shuffled(_base(), DERANGE, parts=("router",)), ref)
    assert "realigned_layers" not in broken
    assert _experts(broken) == _experts(base)


def test_heavily_trained_router_in_standard_order_is_not_reordered(tmp_path, ref):
    g = torch.Generator().manual_seed(8)
    t = _base()
    r = t[P + "gate.weight"]
    noise = torch.randn(r.shape, generator=g)
    t[P + "gate.weight"] = r + noise * (
        r.norm(dim=1, keepdim=True) / noise.norm(dim=1, keepdim=True)
    )
    doc = _fp(tmp_path, "heavy", t, ref)
    assert "realigned_layers" not in doc


def _entry(name, s):
    x = np.zeros(4, dtype=np.float32)
    return dict(
        name=name,
        k=s.shape[0],
        wnorm=float(np.linalg.norm(s)),
        s=base64.b64encode(s.astype(np.float32).tobytes()).decode(),
        x=base64.b64encode(x.tobytes()).decode(),
    )


def _doc(uri, seed, **meta):
    g = np.random.default_rng(seed)
    tensors = [_entry("layers.0.self_attn.q_proj.weight", g.standard_normal((8, 8)))]
    return dict(
        model_uri=uri,
        arch_key="arch",
        ws_version="ws-canon-v2",
        key_id="k1",
        tensors_hash=f"h-{uri}",
        sketch_vec=[0.0] * sketch.VEC_DIM,
        identity_frac={},
        n_tensors=len(tensors),
        secs=0.0,
        tensors=tensors,
        **meta,
    )


class _Bank:
    def __init__(self, banked=()):
        self.banked = {d["model_uri"]: {"status": bank.STATUS_BANK, **d} for d in banked}
        self.put = []
        self.calls = []

    def install(self, monkeypatch):
        monkeypatch.setattr(bank, "find_exact", self.find_exact)
        monkeypatch.setattr(bank, "find_exact_own", lambda doc, ck: None)
        monkeypatch.setattr(bank, "nearest_own", lambda doc, ck, k: [])
        monkeypatch.setattr(bank, "count_scope", lambda doc: len(self.banked))
        monkeypatch.setattr(bank, "root_id", lambda doc: None)
        monkeypatch.setattr(bank, "nearest", self.nearest)
        monkeypatch.setattr(bank, "fetch", self.fetch)
        monkeypatch.setattr(bank, "put_doc", lambda doc, **kw: self.put.append((doc, kw)))

    def find_exact(self, doc, hotkey, coldkey="", block=None):
        self.calls.append(("exact", block))

    def nearest(self, doc, hotkey, k, coldkey="", block=None):
        self.calls.append(("bank", block))
        return [(u, 0.0) for u in self.banked]

    def fetch(self, ids):
        return {i: self.banked[i] for i in ids if i in self.banked}


def _run_gate(monkeypatch, doc, fake, block=500, align=True):
    seen = {}
    fake.install(monkeypatch)
    monkeypatch.setattr(gate, "device", lambda: CPU)
    monkeypatch.setattr(gate, "load_secret", lambda *a: SECRET)
    monkeypatch.setattr(gate, "ref_dir", lambda: "")
    monkeypatch.setattr(gate, "fingerprint", lambda *a, **kw: seen.update(kw) or doc)
    monkeypatch.setattr(gate.config, "DEDUP_ALIGN_EXPERTS", align)
    res = gate.run("dir", doc["model_uri"], "hk-mine", "repo", "dig", "ck", block_number=block)
    return res, seen


def _bank_says(monkeypatch, status, reason):
    monkeypatch.setattr(
        gate,
        "decide",
        lambda cand, bnk, root, identity, th: Verdict(status, reason, next(iter(bnk)), "rules"),
    )


def test_a_realigned_upload_is_judged_by_the_normal_rules_and_noted(monkeypatch):
    fake = _Bank(banked=[_doc("banked", 2, coldkey="ck-x")])
    _bank_says(monkeypatch, "PASS", None)
    doc = _doc("cand", 1, realigned_layers=["layers.0.mlp.", "layers.1.mlp."])
    res, seen = _run_gate(monkeypatch, doc, fake)
    assert res.verdict.status == "PASS" and not res.rejected
    assert res.verdict.notes[0] == "EXPERT ORDER REALIGNED in 2 layer(s)"
    assert seen["align"] == gate.align_params()
    assert fake.put[0][1]["status"] == bank.STATUS_BANK


def test_the_bank_searches_get_the_upload_block(monkeypatch):
    fake = _Bank(banked=[_doc("banked", 2, coldkey="ck-x")])
    _bank_says(monkeypatch, "PASS", None)
    _run_gate(monkeypatch, _doc("cand", 1), fake, block=777)
    assert fake.calls == [("exact", 777), ("bank", 777)]
    assert fake.put[0][1]["block_number"] == 777


def test_alignment_switch_off_restores_the_old_fingerprint(monkeypatch):
    fake = _Bank(banked=[_doc("banked", 2, coldkey="ck-x")])
    _bank_says(monkeypatch, "PASS", None)
    _, seen = _run_gate(monkeypatch, _doc("cand", 1), fake, align=False)
    assert seen["align"] is None


class _Client:
    def __init__(self):
        self.searches, self.indexed = [], []
        self.indices = type("I", (), {"exists": lambda s, index: True})()

    def search(self, index, body):
        self.searches.append(body)
        return {"hits": {"hits": []}}

    def index(self, index, id, body):
        self.indexed.append((id, body))


def test_searches_only_see_models_from_earlier_blocks(monkeypatch):
    c = _Client()
    monkeypatch.setattr(bank, "get_client", lambda: c)
    doc = _doc("cand", 1)
    bank.nearest(doc, "hk", 5, "ck", 700)
    bank.find_exact(doc, "hk", "ck", 700)
    earlier = {"bool": {"must_not": [{"range": {"block_number": {"gte": 700}}}]}}
    knn, exact = c.searches
    assert earlier in knn["query"]["script_score"]["query"]["bool"]["filter"]
    assert earlier in exact["query"]["bool"]["filter"]
    bank.nearest(doc, "hk", 5, "ck")
    bank.find_exact(doc, "hk", "ck")
    assert earlier not in c.searches[-2]["query"]["script_score"]["query"]["bool"]["filter"]
    assert earlier not in c.searches[-1]["query"]["bool"]["filter"]


def test_put_doc_keeps_the_upload_block(monkeypatch):
    c = _Client()
    monkeypatch.setattr(bank, "get_client", lambda: c)
    bank.put_doc(_doc("cand", 1), status=bank.STATUS_BANK, block_number=321)
    assert c.indexed[-1][1]["block_number"] == 321
