from __future__ import annotations

import io
import json
import re
import tarfile
import time

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from albedo_config import RepoContextSettings
from albedo_eval_service.shared.observation_format import (
    OPENHANDS,
    PYTEST_MISSING,
    first_bash_block,
    grounded_observation,
)
from repo_context_service import core
from repo_context_service.command_search import ParseFailure, parse_search, run_search
from repo_context_service.core import (
    _NEGATIVE_TTL_SECONDS,
    _SHA_RULE,
    _TRANSIENT_TTL_SECONDS,
    SCAFFOLDS,
    GroundingContext,
    RepoContextService,
    Scaffold,
    _data_words,
    _editor_view,
    _filter_listing,
    _is_permanent_github_error,
    _NotFound,
    _referenced_paths,
    _safe_name,
    _smith_mirror,
    _SnapshotTooLarge,
    parse_instance,
    scaffold_for,
    source_family,
)

FULL_SHA = "abcdef1234567890abcdef1234567890abcdef12"
SNAPSHOT_KEY = f"o__r__{FULL_SHA[:12]}"


def make_settings(tmp_path, **overrides) -> RepoContextSettings:
    values = {"cache_dir": str(tmp_path / "cache")}
    values.update(overrides)
    return RepoContextSettings(_env_file=None, **values)


def make_service(tmp_path, **overrides) -> RepoContextService:
    return RepoContextService(make_settings(tmp_path, **overrides))


def make_snapshot(service, files: dict[str, str], listing: list[str] | None = None):
    snapshot = service.cache_dir / "snapshots" / SNAPSHOT_KEY
    for rel, text in files.items():
        target = snapshot / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    (snapshot / ".albedo-listing.json").write_text(json.dumps(listing or sorted(files)))
    (snapshot / ".albedo-repo-context-done").write_text("")
    return snapshot


def test_cache_dir_is_required(tmp_path):
    with pytest.raises(ValueError):
        RepoContextService(RepoContextSettings(_env_file=None, cache_dir=""))


def test_parse_instance_formats():
    swe = parse_instance("swe-zero", "azure__secrets-store-csi-driver-provider-azure-466")
    assert swe.owner == "azure"
    assert (swe.repo, swe.pr) == ("secrets-store-csi-driver-provider-azure", "466")
    mini = parse_instance("mini-coder", "seperman__deepdiff.4b8fa12__m1")
    assert (mini.owner, mini.repo, mini.commit) == ("seperman", "deepdiff", "4b8fa12")
    goat = parse_instance("mini-coder", "arp242__goatcounter.854b1dd2.lm_modify__1vcxllzm")
    assert (goat.owner, goat.repo, goat.commit) == ("arp242", "goatcounter", "854b1dd2")
    rs = parse_instance("mini-coder-rs", "arp242__goatcounter.854b1dd2.lm_modify__1vcxllzm")
    assert (rs.owner, rs.repo, rs.commit) == ("arp242", "goatcounter", "854b1dd2")
    hero = parse_instance("swe-hero", "pandas-dev__pandas-dbf8aaf4a3f3b41e5c1a402473df5da43813948f")
    assert (hero.owner, hero.repo, hero.pr) == ("pandas-dev", "pandas", None)
    assert hero.commit == "dbf8aaf4a3f3b41e5c1a402473df5da43813948f"
    ost = parse_instance("open-swe-traces", "python-attrs__attrs-770")
    assert (ost.owner, ost.repo, ost.pr) == ("python-attrs", "attrs", "770")
    # the same source also ships `owner_repo_pr<N>`; without this it grounds as kind=trajectory
    pull = parse_instance("open-swe-traces", "fillipe-gsm_python-tsp_pr39")
    assert (pull.owner, pull.repo, pull.pr) == ("fillipe-gsm", "python-tsp", "39")
    # owner is whatever precedes the FIRST underscore, so hyphenated repos survive
    hyphen = parse_instance("open-swe-traces", "sphinx-contrib_confluencebuilder_pr1093")
    assert (hyphen.owner, hyphen.repo, hyphen.pr) == ("sphinx-contrib", "confluencebuilder", "1093")
    assert parse_instance("swe-zero", "owner__repo-notanumber") is None
    assert parse_instance("mini-coder", "noseparator") is None
    assert parse_instance("mini-coder", "owner__repo.ZZZZZZ") is None


def test_scale_swe_machines_have_their_own_scaffold():
    rebench, scale = "owner__repo-12", "owner_repo_pr12"
    v10 = scaffold_for("open-swe-traces-v1.0", "openhands", rebench)
    assert v10 is SCAFFOLDS[("open-swe-traces", "openhands")] and v10.detached
    oh = scaffold_for("open-swe-traces-v1.1", "openhands", scale)
    assert (oh.branch, oh.detached) == ("scaleswe", False)
    assert scaffold_for("open-swe-traces-v1.1", "swe_agent", scale).detached_at
    mswea = scaffold_for("open-swe-traces-v1.1", "returncode", rebench)
    assert (mswea.shell, mswea.detached, mswea.detached_at) == ("dash", True, False)
    assert scaffold_for("open-swe-traces-v1.1", "returncode", scale).detached_at
    v12 = scaffold_for("open-swe-traces-v1.2", "returncode", scale)
    assert (v12.branch, v12.detached) == ("scaleswe", False)
    assert scaffold_for("mini-coder", "returncode", "a__b.1234567.pr_1").shell == "bash"
    assert scaffold_for("unknown-source", "openhands") == Scaffold()


def test_affine_machines_have_their_own_scaffolds():
    rebench, scale, smith = "owner__repo-12", "owner_repo_pr12", "a__b.1234567.lm_modify__x"
    oh = scaffold_for("affine-openhands", "returncode", rebench)
    assert (oh.columns, oh.decorate, oh.detached, oh.editor) == (core._ANY_WIDTH, True, False, "")
    shells = {
        "mswea": ("/bin/sh: 1: ", "dash"),
        "bash": ("bash: line 1: ", "bash"),
        "tools": ("/bin/bash: line 1: ", "bash"),
    }
    for machine, (errors, shell) in shells.items():
        pr = scaffold_for(f"affine-{machine}", "returncode", rebench)
        assert (pr.errors, pr.shell, pr.detached, pr.branch) == (errors, shell, False, "")
        on_branch = scaffold_for(f"affine-{machine}", "returncode", scale)
        assert (on_branch.errors, on_branch.branch, on_branch.detached) == (
            errors,
            "scaleswe",
            False,
        )
        mirror = scaffold_for(f"mini-coder-affine-{machine}", "returncode", smith)
        assert (mirror.errors, mirror.shell) == (errors, shell)


def test_a_versioned_source_shares_its_family_scaffold():
    assert source_family("open-swe-traces-v1.0") == "open-swe-traces"
    assert source_family("open-swe-traces-v1.2") == "open-swe-traces"
    assert source_family("mini-coder-rs") == "mini-coder-rs"
    assert source_family("swe-hero") == "swe-hero"
    for fmt in ("openhands", "swe_agent"):
        assert (
            SCAFFOLDS.get((source_family("open-swe-traces-v1.1"), fmt))
            is SCAFFOLDS[("open-swe-traces", fmt)]
        )


def test_grounding_and_the_replay_read_the_command_the_judge_reads(tmp_path, monkeypatch):
    """The judge runs the first `bash` fence or `<..._bash_...>` tag; grounding must answer that
    command and the replay must apply it. A code snippet in an untagged or another language's
    fence is not a command: replaying `rm` out of it would change a checkout nothing changed."""
    service = make_service(tmp_path)
    make_snapshot(service, {"a.py": "alpha\n", "b.py": "beta\n"})
    monkeypatch.setattr(service, "_resolve_sha", lambda ref: ("o", "r", FULL_SHA))
    done = {"role": "user", "content": "<returncode>0</returncode>\n<output>\n</output>"}
    tagged = "<mswea_bash_command>cat a.py</mswea_bash_command>"
    assert first_bash_block(tagged) == "cat a.py"
    result = service.repo_context_for_instance("swe-zero", "o__r-12", tagged, fmt="returncode")
    assert result.exact_output == "alpha"
    removed = [{"role": "assistant", "content": "<mswea_bash_command>rm a.py</mswea_bash_command>"}]
    after = service.repo_context_for_instance(
        "swe-zero", "o__r-12", "```bash\ncat a.py\n```", removed + [done], fmt="returncode"
    )
    assert after.exact_output == "cat: a.py: No such file or directory"
    snippet = [{"role": "assistant", "content": "Next:\n```\nrm b.py\n```"}, done]
    kept = service.repo_context_for_instance(
        "swe-zero", "o__r-12", "```bash\ncat b.py\n```", snippet, fmt="returncode"
    )
    assert kept.exact_output == "beta"


def test_iid_lookup_from_manifest(tmp_path):
    manifest = {
        "sources": [
            {
                "name": "swe-zero",
                "shards": [
                    {
                        "path": "swe-zero/data/train-00000.parquet",
                        "rows": 2,
                        "rows_meta": [{"iid": "o__r-1", "asst": 3}, {"iid": "o2__r2-5", "asst": 4}],
                    }
                ],
            }
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    service = make_service(tmp_path, dataset_manifest_path=str(manifest_path))
    assert service._iid_for("swe-zero/data/train-00000.parquet", 1) == ("swe-zero", "o2__r2-5")
    assert service._iid_for("swe-zero/data/train-00000.parquet", 9) is None
    assert service._iid_for("mini-coder/data/train-00000.parquet", 0) is None

    bad_hash = make_service(
        tmp_path, dataset_manifest_path=str(manifest_path), dataset_manifest_hash="0" * 64
    )
    assert bad_hash._iid_for("swe-zero/data/train-00000.parquet", 1) is None


def test_resolve_sha_renamed_repo_and_cache(tmp_path, monkeypatch):
    service = make_service(tmp_path)
    calls = []

    def fake_github_json(path):
        calls.append(path)
        if path == "/repos/old/name/pulls/12":
            raise _NotFound(path)
        if path == "/repos/old/name":
            return {"full_name": "new/name"}
        if path == "/repos/new/name/pulls/12":
            return {"base": {"sha": FULL_SHA}}
        raise AssertionError(path)

    monkeypatch.setattr(service, "_github_json", fake_github_json)
    ref = parse_instance("swe-zero", "old__name-12")
    assert service._resolve_sha(ref) == ("new", "name", FULL_SHA)
    assert service._resolve_sha(ref) == ("new", "name", FULL_SHA)
    assert len(calls) == 3


def test_resolve_sha_negative_cache_ttl(tmp_path, monkeypatch):
    service = make_service(tmp_path)
    calls = []

    def failing(path):
        calls.append(path)
        raise RuntimeError("boom")

    monkeypatch.setattr(service, "_github_json", failing)
    ref = parse_instance("swe-zero", "own__repo-3")
    assert service._resolve_sha(ref) is None
    assert service._resolve_sha(ref) is None
    assert len(calls) == 1

    cache_file = service.cache_dir / "shas" / "own__repo-3.json"
    stale = json.loads(cache_file.read_text())
    stale["failed_at"] = time.time() - 2 * 24 * 3600
    cache_file.write_text(json.dumps(stale))
    assert service._resolve_sha(ref) is None
    assert len(calls) == 2


def test_is_permanent_github_error_classification():
    assert _is_permanent_github_error(_NotFound("/x")) is True
    for code in (400, 410, 422, 451):
        resp = httpx.Response(code, request=httpx.Request("GET", "https://api.github.com/x"))
        err = httpx.HTTPStatusError("e", request=resp.request, response=resp)
        assert _is_permanent_github_error(err)
    # rate-limit / transport errors are NOT permanent -> only short transient caching
    assert _is_permanent_github_error(RuntimeError("github 403 for /x")) is False
    assert _is_permanent_github_error(httpx.ConnectError("boom")) is False


def test_resolve_sha_transient_failure_uses_short_ttl(tmp_path, monkeypatch):
    """A rate-limit/transport failure is cached only for the transient TTL, so a throttled or
    unauthenticated run self-heals on the next pass instead of staying dead for 24h."""
    service = make_service(tmp_path)
    calls = []

    def failing(path):
        calls.append(path)
        raise RuntimeError("github 403 for " + path)

    monkeypatch.setattr(service, "_github_json", failing)
    ref = parse_instance("swe-zero", "own__repo-3")
    assert service._resolve_sha(ref) is None
    cache_file = service.cache_dir / "shas" / "own__repo-3.json"
    assert json.loads(cache_file.read_text())["kind"] == "transient"

    # aged just past the transient TTL but far inside the 24h negative TTL -> must retry
    entry = json.loads(cache_file.read_text())
    entry["failed_at"] = time.time() - (_TRANSIENT_TTL_SECONDS + 60)
    cache_file.write_text(json.dumps(entry))
    assert service._resolve_sha(ref) is None
    assert len(calls) == 2


def test_resolve_sha_permanent_failure_uses_long_ttl(tmp_path, monkeypatch):
    """A genuine 404 (PR/commit gone, rename lookup also 404) stays cached for the full negative
    TTL and is not retried on every request."""
    service = make_service(tmp_path)
    calls = []

    def failing(path):
        calls.append(path)
        raise _NotFound(path)

    monkeypatch.setattr(service, "_github_json", failing)
    ref = parse_instance("swe-zero", "gone__repo-9")
    assert service._resolve_sha(ref) is None
    cache_file = service.cache_dir / "shas" / "gone__repo-9.json"
    assert json.loads(cache_file.read_text())["kind"] == "permanent"
    settled = len(calls)

    # aged past the transient TTL but within the 24h negative TTL -> still suppressed
    entry = json.loads(cache_file.read_text())
    entry["failed_at"] = time.time() - (_TRANSIENT_TTL_SECONDS + 3600)
    cache_file.write_text(json.dumps(entry))
    assert service._resolve_sha(ref) is None
    assert len(calls) == settled

    # aged past the full negative TTL -> retry
    entry["failed_at"] = time.time() - (_NEGATIVE_TTL_SECONDS + 60)
    cache_file.write_text(json.dumps(entry))
    assert service._resolve_sha(ref) is None
    assert len(calls) > settled


def test_resolve_sha_legacy_failure_entry_self_heals(tmp_path, monkeypatch):
    """Failure entries written before ``kind`` existed (the poisoned ones from the unauthenticated
    run) expire at the transient TTL, so a fixed/authenticated rerun re-resolves them."""
    service = make_service(tmp_path)
    cache_dir = service.cache_dir / "shas"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "own__repo-5.json").write_text(
        json.dumps({"error": "boom", "failed_at": time.time() - (_TRANSIENT_TTL_SECONDS + 60)})
    )

    calls = []

    def ok(path):
        calls.append(path)
        return {"base": {"sha": FULL_SHA}}

    monkeypatch.setattr(service, "_github_json", ok)
    ref = parse_instance("swe-zero", "own__repo-5")
    assert service._resolve_sha(ref) == ("own", "repo", FULL_SHA)
    assert calls  # retried instead of honoring the stale legacy failure


def _tar_bytes() -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:

        def add(name, data=b"", **attrs):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            for key, value in attrs.items():
                setattr(info, key, value)
            tar.addfile(info, io.BytesIO(data) if info.isreg() else None)

        add("repo-abc/src/app.py", b"print('hi')\n")
        add("repo-abc/link.py", type=tarfile.SYMTYPE, linkname="/etc/passwd")
        add("repo-abc/../evil.txt", b"evil")
        add("/abs.txt", b"evil")
        big = tarfile.TarInfo("repo-abc/huge.bin")
        big.size = 3 * 1024 * 1024
        tar.addfile(big, io.BytesIO(b"0" * big.size))
    return buffer.getvalue()


def test_extract_tarball_sanitizes_members(tmp_path):
    service = make_service(tmp_path)
    tar_path = tmp_path / "snap.tar.gz"
    tar_path.write_bytes(_tar_bytes())
    dest = tmp_path / "out"
    dest.mkdir()
    listing, extracted_bytes = service._extract_tarball(tar_path, dest)
    # a link and an oversized file are listed, with text not kept
    assert listing == ["huge.bin", "link.py", "src/app.py"]
    assert extracted_bytes == len(b"print('hi')\n")
    assert (dest / "src/app.py").read_text() == "print('hi')\n"
    assert not (tmp_path / "evil.txt").exists()
    assert not (dest / "link.py").exists()
    assert not (dest / "huge.bin").exists()


def test_repo_block_contents_and_containment(tmp_path, monkeypatch):
    service = make_service(tmp_path)
    snapshot = make_snapshot(
        service,
        {"src/app.py": "APP CONTENT\n", "README.md": "readme\n"},
        listing=["README.md", "src/app.py", "link.py"],
    )
    (snapshot / "link.py").symlink_to("/etc/passwd")
    monkeypatch.setattr(service, "_resolve_sha", lambda ref: ("o", "r", FULL_SHA))

    command = (
        "```bash\nfind . -type f && cat src/app.py .env /etc/passwd ../../secret missing/file.py "
        "&& sed -n s/a/b/ src/app.py && grep -r foo src/\n```"
    )
    result = service.repo_context_for_instance("swe-zero", "o__r-12", command)
    assert result.kind == "repo"
    block = result.context
    assert "APP CONTENT" in block
    assert "./src/app.py" in block
    not_present = block.split("FILES NOT PRESENT")[1]
    assert "- .env" in not_present
    # outside the checkout: not the repo's to deny
    assert "- /etc/passwd" not in not_present and "- ../../secret" not in not_present
    assert "- missing/file.py" in not_present
    assert "s/a/b" not in not_present
    assert "- src" not in not_present
    assert "- ." not in not_present.split("- .env")[0]
    assert service._read_snapshot_file(snapshot, "link.py") is None
    assert "root:" not in block


def test_commands_cannot_reach_other_snapshots_or_host(tmp_path, monkeypatch):
    service = make_service(tmp_path)
    make_snapshot(service, {"src/app.py": "APP CONTENT\n"}, listing=["src/app.py"])
    other = service.cache_dir / "snapshots" / "other__repo__000000000000"
    (other / "conf").mkdir(parents=True)
    (other / "conf/secret.py").write_text("OTHER SNAPSHOT SECRET\n")
    (service.cache_dir / "shas").mkdir(parents=True, exist_ok=True)
    (service.cache_dir / "shas" / "token.json").write_text('{"sha": "HOST SECRET"}')
    monkeypatch.setattr(service, "_resolve_sha", lambda ref: ("o", "r", FULL_SHA))

    command = (
        "```bash\ncat ../other__repo__000000000000/conf/secret.py conf/secret.py "
        "../../shas/token.json /etc/passwd\n```"
    )
    block = service.repo_context_for_instance("swe-zero", "o__r-1", command).context
    assert "OTHER SNAPSHOT SECRET" not in block
    assert "HOST SECRET" not in block
    assert "root:" not in block
    snapshot = service.cache_dir / "snapshots" / SNAPSHOT_KEY
    escape = "../other__repo__000000000000/conf/secret.py"
    assert service._read_snapshot_file(snapshot, escape) is None
    assert service._read_snapshot_file(snapshot, "../../shas/token.json") is None


def test_filter_listing_and_caps(tmp_path, monkeypatch):
    listing = [f"pkg/mod_{i}.py" for i in range(5)] + ["docs/guide.md"]
    kept, filtered = _filter_listing(listing, "find . -name '*.py'")
    assert filtered is True
    assert kept == [f"pkg/mod_{i}.py" for i in range(5)]

    service = make_service(tmp_path, max_paths=3)
    make_snapshot(service, {p: "x" for p in listing}, listing=listing)
    monkeypatch.setattr(service, "_resolve_sha", lambda ref: ("o", "r", FULL_SHA))
    result = service.repo_context_for_instance(
        "swe-zero", "o__r-1", "```bash\nfind $(pwd) -name '*.py'\n```"
    )
    assert "... (+2 more matching files)" in result.context

    nothing = service.repo_context_for_instance(
        "swe-zero", "o__r-1", "```bash\nfind $(pwd) -name '*.zig'\n```"
    )
    assert nothing.kind == "repo"
    assert "no files in this repository match" in nothing.context
    assert "./pkg/mod_0.py" not in nothing.context


def test_a_ranged_view_piped_through_cat_n_is_computable():
    listing = ["a.py"]
    read = {"a.py": "l1\nl2\nl3\nl4\nl5\n"}.get
    out = run_search(parse_search("sed -n '2,4p' a.py | cat -n"), read, listing).output
    assert out == "     1\tl2\n     2\tl3\n     3\tl4"
    assert isinstance(parse_search("cat a.py | cat -A"), ParseFailure)


def test_a_quoted_pattern_is_not_read_as_shell_syntax():
    assert not isinstance(parse_search('grep -n "a || b" a.py'), ParseFailure)
    assert not isinstance(parse_search("grep -n 'x; y' a.py"), ParseFailure)
    assert isinstance(parse_search("cat a.py; rm -rf /"), ParseFailure)


def test_a_long_listing_is_derived_from_the_snapshot():
    listing = ["a.py", "pkg/b.py"]
    read = {"a.py": "L1\nL2\n", "pkg/b.py": "X\n"}.get
    rows = run_search(parse_search("ls -l"), read, listing).output.split("\n")
    assert rows[0] == "total 8"
    # each column as wide as its widest value, as GNU ls aligns them
    assert re.fullmatch(r"-rw-r--r-- 1 root root    6 \w{3} +\d+ [\d:]+ a\.py", rows[1]), rows[1]
    assert re.fullmatch(r"drwxr-xr-x 2 root root 4096 \w{3} +\d+ [\d:]+ pkg", rows[2]), rows[2]
    assert run_search(parse_search("ls -l a.py"), read, listing).output.endswith(" a.py")
    assert isinstance(parse_search("ls -lh"), ParseFailure)


_FIND_TREE = [
    "docs/guide.md",
    "node_modules/x/y.py",
    "src/a.py",
    "src/util/b.py",
    "src/util/deep/c.py",
    "tests/t_one.py",
]


def _find(cmd: str) -> list[str]:
    plan = parse_search(cmd)
    assert not isinstance(plan, ParseFailure), f"{cmd} -> {plan}"
    return run_search(plan, {p: "x\n" for p in _FIND_TREE}.get, _FIND_TREE).output.split("\n")


def _declined(cmd: str) -> bool:
    plan = parse_search(cmd)
    return isinstance(plan, ParseFailure) or isinstance(
        run_search(plan, {p: "x\n" for p in _FIND_TREE}.get, _FIND_TREE), ParseFailure
    )


def test_find_answers_without_needing_a_grep_downstream():
    assert _find("find . -name '*.py'") == [
        "./node_modules/x/y.py",
        "./src/a.py",
        "./src/util/b.py",
        "./src/util/deep/c.py",
        "./tests/t_one.py",
    ]
    # the order of a walk is the filesystem's, so what `head` keeps of it is not known
    assert _declined("find . -name '*.py' | head -2")
    assert _find("find . -name '*.py' | sort | head -2") == ["./node_modules/x/y.py", "./src/a.py"]
    assert _find("find . -name '*.py' | wc -l") == ["5"]
    assert _find("find . -name '*.py' | grep util") == ["./src/util/b.py", "./src/util/deep/c.py"]


def test_find_depth_and_directory_predicates():
    assert _find("find src -type d") == ["src", "src/util", "src/util/deep"]
    # the checkout's .git holds directories whose names are not known
    assert _declined("find . -type d")
    assert _find("find . -maxdepth 1 -type d") == [
        ".",
        "./.git",
        "./docs",
        "./node_modules",
        "./src",
        "./tests",
    ]
    assert _find("find src -maxdepth 2 -name '*.py'") == ["src/a.py", "src/util/b.py"]
    assert _find("find . -name 'util' -type d") == ["./src/util"]


def test_find_excludes_negated_and_pruned_subtrees():
    expected = ["./src/a.py", "./src/util/b.py", "./src/util/deep/c.py", "./tests/t_one.py"]
    assert _find("find . -name '*.py' -not -path '*/node_modules/*'") == expected
    assert _find("find . -path './node_modules' -prune -o -name '*.py' -print") == expected


def test_find_or_is_honoured_only_within_one_predicate():
    assert _find("find . -name '*.py' -o -name '*.md'") == [
        "./docs/guide.md",
        "./node_modules/x/y.py",
        "./src/a.py",
        "./src/util/b.py",
        "./src/util/deep/c.py",
        "./tests/t_one.py",
    ]
    assert _find("find src -path '*util*' -o -name 'a.py'") == [
        "src/a.py", "src/util", "src/util/b.py", "src/util/deep", "src/util/deep/c.py"
    ]  # fmt: skip
    # `-o` next to another test binds tighter than it looks: not reproduced
    assert _declined("find . -name '*.py' -o -name '*.md' -type f")


def test_find_exec_ls_renders_one_long_row_per_hit():
    assert _find("find . -name '*.py' -exec ls -la {} \\;") == [
        "-rw-r--r-- 1 root root 2 Jan  3 20:00 ./node_modules/x/y.py",
        "-rw-r--r-- 1 root root 2 Jan  3 20:00 ./src/a.py",
        "-rw-r--r-- 1 root root 2 Jan  3 20:00 ./src/util/b.py",
        "-rw-r--r-- 1 root root 2 Jan  3 20:00 ./src/util/deep/c.py",
        "-rw-r--r-- 1 root root 2 Jan  3 20:00 ./tests/t_one.py",
    ]
    assert _find("find src -type f -exec ls {} \\;") == [
        "src/a.py",
        "src/util/b.py",
        "src/util/deep/c.py",
    ]


def test_find_still_refuses_what_it_cannot_stand_behind():
    for cmd in (
        "find . -name '*.py' | awk '{print $1}'",
        "find . -name '*.py' -exec cat {} \\;",
        "find . -name '*.py' -exec ls -lh {} \\;",
        "find . -type d -exec ls -la {} \\;",
        "find $(pwd) -name '*.py'",
        "find . -newer setup.py",
    ):
        assert isinstance(parse_search(cmd), ParseFailure), cmd


def test_a_missing_read_target_reports_the_shell_error_it_would_print():
    listing = ["a.py"]
    read = {"a.py": "l1\n"}.get
    expected = {
        "cat gone.py": "cat: gone.py: No such file or directory",
        "nl -ba gone.py": "nl: gone.py: No such file or directory",
        "head -5 gone.py": "head: cannot open 'gone.py' for reading: No such file or directory",
        "tail -5 gone.py": "tail: cannot open 'gone.py' for reading: No such file or directory",
        "sed -n '1,3p' gone.py": "sed: can't read gone.py: No such file or directory",
    }
    for cmd, message in expected.items():
        result = run_search(parse_search(cmd), read, listing)
        assert not isinstance(result, ParseFailure), cmd
        assert result.output == message
        assert result.missing == ["gone.py"]
    assert run_search(parse_search("cat a.py"), read, listing).output == "l1"


def test_grep_reports_a_directory_operand_as_a_file_without_matches():
    """GNU grep prints `Is a directory` and then counts or lists the directory as it would an
    empty file: `-c` gives it 0, `-L` names it, `-l` and plain matching print nothing for it."""
    listing = ["a.py", "b.cfg", "pkg/a.py"]
    read = {"a.py": "S here\n", "b.cfg": "x\n", "pkg/a.py": "y\n"}.get

    def out(cmd):
        return run_search(parse_search(cmd), read, listing)

    counted = out("grep -c S a.py pkg")
    assert (counted.output, counted.returncode) == ("a.py:1\ngrep: pkg: Is a directory\npkg:0", 2)
    assert out("grep -c S pkg").output == "grep: pkg: Is a directory\n0"
    assert out("grep -s -c S b.cfg pkg").output == "b.cfg:0\npkg:0"
    assert out("grep -L S b.cfg pkg a.py").output == "b.cfg\ngrep: pkg: Is a directory\npkg"
    assert out("grep -l S b.cfg pkg a.py").output == "grep: pkg: Is a directory\na.py"
    assert out("grep -n S a.py pkg").output == "a.py:1:S here\ngrep: pkg: Is a directory"
    assert out("grep -c S nothere b.cfg").output == (
        "grep: nothere: No such file or directory\nb.cfg:0"
    )


def test_ls_reports_missing_operands_in_argument_order_before_the_sorted_listing():
    listing = ["a.py", "b.cfg", "adir/q", "zdir/w"]
    read = {path: "x\n" for path in listing}.get
    result = run_search(parse_search("ls zmiss b.cfg amiss zdir adir a.py"), read, listing)
    assert result.output == (
        "ls: cannot access 'zmiss': No such file or directory\n"
        "ls: cannot access 'amiss': No such file or directory\n"
        "a.py\nb.cfg\n\nadir:\nq\n\nzdir:\nw"
    )
    assert result.returncode == 2


def test_head_prints_a_header_before_a_directory_error_among_several_operands():
    """head opens a directory before the read fails, so its `==> name <==` header is printed;
    a missing operand gets no header, and a blank line precedes every header but the first."""
    listing = ["a.py", "adir/q", "tests/t.py"]
    read = {"a.py": "S here\n", "adir/q": "q\n", "tests/t.py": "x\n"}.get

    def out(cmd):
        return run_search(parse_search(cmd), read, listing).output

    assert out("head -1 nothere tests") == (
        "head: cannot open 'nothere' for reading: No such file or directory\n"
        "==> tests <==\nhead: error reading 'tests': Is a directory"
    )
    assert out("head -1 tests adir") == (
        "==> tests <==\nhead: error reading 'tests': Is a directory\n\n"
        "==> adir <==\nhead: error reading 'adir': Is a directory"
    )
    assert out("head -n 1 tests nothere a.py") == (
        "==> tests <==\nhead: error reading 'tests': Is a directory\n"
        "head: cannot open 'nothere' for reading: No such file or directory\n\n"
        "==> a.py <==\nS here"
    )
    assert out("head -1 tests") == "head: error reading 'tests': Is a directory"


def test_a_path_under_a_file_is_not_a_directory():
    listing = ["a.py", "sub"]
    read = {"a.py": "S here\n", "sub": "x\n"}.get
    expected = {
        "cat sub/conf.cfg": "cat: sub/conf.cfg: Not a directory",
        "cat sub/a/b": "cat: sub/a/b: Not a directory",
        "nl -ba sub/x": "nl: sub/x: Not a directory",
        "head -1 sub/x": "head: cannot open 'sub/x' for reading: Not a directory",
        "tail -n 1 sub/x": "tail: cannot open 'sub/x' for reading: Not a directory",
        "sed -n 1p sub/x": "sed: can't read sub/x: Not a directory",
        "wc -l sub/x": "wc: sub/x: Not a directory",
        "grep S sub/x": "grep: sub/x: Not a directory",
        "grep -c S sub/x a.py": "grep: sub/x: Not a directory\na.py:1",
        "find sub/x -type f": "find: 'sub/x': Not a directory",
        "ls sub/x": "ls: cannot access 'sub/x': Not a directory",
        "head -1 sub/x a.py": (
            "head: cannot open 'sub/x' for reading: Not a directory\n==> a.py <==\nS here"
        ),
    }
    for cmd, message in expected.items():
        result = run_search(parse_search(cmd), read, listing)
        assert not isinstance(result, ParseFailure), cmd
        assert result.output == message, cmd


def test_grep_flags_that_change_nothing_on_text_are_accepted():
    listing = ["a.py"]
    read = {"a.py": "alpha\nbeta\n"}.get
    assert run_search(parse_search("grep -an alpha a.py"), read, listing).output == "1:alpha"
    assert run_search(parse_search("grep -n alpha a.py | cat"), read, listing).output == "1:alpha"
    assert isinstance(parse_search("grep -n alpha a.py | cat -v"), ParseFailure)


def test_a_pattern_grep_itself_rejects_is_reported_not_refused():
    listing = ["a.py"]
    read = {"a.py": "alpha\n"}.get
    assert (
        run_search(parse_search('grep -n "self.get\\(" a.py'), read, listing).output
        == "grep: Unmatched ( or \\("
    )
    assert (
        run_search(parse_search('grep -rn "reverse\\|[::-1]" .'), read, listing).output
        == "grep: Invalid range end"
    )
    quiet = run_search(parse_search('grep -rn "reverse\\|[::-1]" . 2>/dev/null'), read, listing)
    assert quiet.output == "" and quiet.empty is True
    assert isinstance(parse_search('rg -n "self.get\\(" a.py'), ParseFailure)


def test_paths_built_at_container_setup_are_not_claimed_absent():
    listing = ["a.py"]
    read = {"a.py": "l1\n"}.get
    for cmd in (
        "cat node_modules/.bin/jest",
        "head -5 build/build_config.json",
        "cat faker/providers/ssn.pyc",
        "head -20 .venv/lib/python3.11/site.py",
    ):
        assert isinstance(run_search(parse_search(cmd), read, listing), ParseFailure), cmd


def test_echo_prints_its_argument_and_refuses_escapes():
    assert run_search(parse_search("echo hello world"), {}.get, []).output == "hello world"
    assert run_search(parse_search('echo "a  b"'), {}.get, []).output == "a  b"
    assert isinstance(parse_search("echo -e 'a\\nb'"), ParseFailure)


def test_output_sent_to_a_file_is_not_reported_as_printed():
    for cmd in (
        "cat a.py > out.txt",
        "ls -l > listing.txt",
        "grep -n L a.py >> hits.txt",
        "echo hi > f.txt",
    ):
        assert isinstance(parse_search(cmd), ParseFailure), cmd
    assert not isinstance(parse_search("grep -n L a.py 2>/dev/null"), ParseFailure)


def test_renderable_chain_stages_are_attached_as_evidence(tmp_path, monkeypatch):
    service = make_service(tmp_path)
    make_snapshot(service, {"a.py": "L1\nL2\nL3\n"})
    monkeypatch.setattr(service, "_resolve_sha", lambda ref: ("o", "r", FULL_SHA))

    mixed = service.repo_context_for_instance(
        "swe-zero", "o__r-1", "```bash\nsed -n '1,2p' a.py && python run.py\n```"
    )
    assert "CHAIN STAGES —" in mixed.context
    assert "$ sed -n '1,2p' a.py\nL1\nL2" in mixed.context
    assert mixed.exact_output is None
    assert not mixed.context.lstrip().startswith("COMMAND OUTPUT —")

    nothing = service.repo_context_for_instance(
        "swe-zero", "o__r-1", "```bash\ncd /testbed && python run.py\n```"
    )
    assert "CHAIN STAGES —" not in nothing.context


def test_exact_output_is_offered_only_when_it_can_be_stood_behind(tmp_path, monkeypatch):
    service = make_service(tmp_path)
    make_snapshot(service, {"src/app.py": "APP CONTENT\n"})
    monkeypatch.setattr(service, "_resolve_sha", lambda ref: ("o", "r", FULL_SHA))

    computed = service.repo_context_for_instance(
        "swe-zero", "o__r-1", "```bash\ncat src/app.py\n```"
    )
    assert computed.exact_output == "APP CONTENT"

    listing_only = service.repo_context_for_instance(
        "swe-zero", "o__r-1", "```bash\npython src/app.py\n```"
    )
    assert listing_only.context is not None
    assert listing_only.exact_output is None


def test_contents_survive_huge_listing(tmp_path, monkeypatch):
    listing = [f"pkg/module_{i:05}.py" for i in range(6000)]
    service = make_service(tmp_path, max_context_chars=30000)
    make_snapshot(
        service, {"src/target.py": "TARGET CONTENT " * 100}, listing=listing + ["src/target.py"]
    )
    monkeypatch.setattr(service, "_resolve_sha", lambda ref: ("o", "r", FULL_SHA))
    block = service.repo_context_for_instance(
        "swe-zero", "o__r-1", "```bash\npython src/target.py\n```"
    ).context
    assert len(block) <= 30000 + len("\n... (truncated)")
    assert "TARGET CONTENT" in block
    assert "more matching files)" in block
    assert block.index("TARGET CONTENT") > block.index("more matching files")


def _write_trajectory_shard(root, turns):
    shard = root / "swe-zero" / "data" / "train-00000.parquet"
    shard.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({"instance_id": ["o__r-1"], "messages": [turns]})
    pq.write_table(table, shard)


def test_iid_resolved_from_parquet_without_manifest(tmp_path):
    root = tmp_path / "dataset"
    _write_trajectory_shard(root, [{"role": "user", "content": "task"}])
    service = make_service(tmp_path, dataset_root=str(root))
    assert service._iid_for("swe-zero/data/train-00000.parquet", 0) == ("swe-zero", "o__r-1")
    assert service._iid_for("swe-zero/data/train-00000.parquet", 9) is None
    assert service._iid_for("swe-zero/data/train-99999.parquet", 0) is None


def test_trajectory_fallback_when_repo_unavailable(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    _write_trajectory_shard(
        root,
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "THOUGHT: a\n\n```bash\nls src\n```"},
            {"role": "user", "content": "Observation: obs-1"},
            {"role": "assistant", "content": "```bash\ncat src/x.py\n```"},
            {"role": "user", "content": "Observation: obs-2"},
            {
                "role": "assistant",
                "content": "```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
                " && git add -A && git diff --cached\n```",
            },
            {"role": "user", "content": "Observation: GOLD SOLUTION DIFF"},
        ],
    )
    service = make_service(tmp_path, dataset_root=str(root))
    monkeypatch.setattr(
        service,
        "repo_context_for_instance",
        lambda *a, **k: GroundingContext(context=None, kind="none", reason="snapshot_unavailable"),
    )
    result = service.context_for("swe-zero/data/train-00000.parquet:0:0", "```bash\npwd\n```")
    assert result.kind == "trajectory"
    assert "REFERENCE STEP 1" in result.context
    assert "obs-1" in result.context and "obs-2" in result.context
    assert "THOUGHT" not in result.context
    assert "GOLD SOLUTION DIFF" not in result.context
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" not in result.context

    later = service.context_for("swe-zero/data/train-00000.parquet:0:1", "```bash\npwd\n```")
    assert "obs-1" not in later.context and "obs-2" in later.context


def test_fallback_chain_to_none_and_never_raises(tmp_path, monkeypatch):
    service = make_service(tmp_path)
    result = service.context_for("swe-zero/data/train-00000.parquet:0:0", "```bash\nls\n```")
    assert result == GroundingContext(context=None, kind="none", reason="iid_unresolved")

    assert service.context_for("not a sample id", "x").kind == "none"

    monkeypatch.setattr(
        service, "_context_for", lambda *a: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    assert service.context_for("swe-zero/data/train-00000.parquet:0:0", "x").kind == "none"


def test_no_bypass_even_with_poisoned_listing(tmp_path, monkeypatch):
    service = make_service(tmp_path)
    host_secret = tmp_path / "host_secret.txt"
    host_secret.write_text("HOST SECRET\n")
    poisoned = [
        "../../../host_secret.txt",
        str(host_secret),
        "evil/host_secret.txt",
        "src/app.py",
    ]
    snapshot = make_snapshot(service, {"src/app.py": "APP CONTENT\n"}, listing=poisoned)
    (snapshot / "evil").symlink_to(tmp_path)
    monkeypatch.setattr(service, "_resolve_sha", lambda ref: ("o", "r", FULL_SHA))

    command = (
        f"```bash\ncat ../../../host_secret.txt evil/host_secret.txt src/app.py {host_secret}\n```"
    )
    block = service.repo_context_for_instance("swe-zero", "o__r-1", command).context
    assert "HOST SECRET" not in block
    assert "APP CONTENT" in block
    for entry in poisoned[:3]:
        assert service._read_snapshot_file(snapshot, entry) is None


def test_extract_tarball_skips_metadata_collisions(tmp_path):
    service = make_service(tmp_path)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, data in [
            ("repo-abc/.albedo-listing.json", b'["fake/entry.py"]'),
            ("repo-abc/.albedo-repo-context-done", b'{"bytes": 0}'),
            ("repo-abc/real.py", b"ok"),
        ]:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    tar_path = tmp_path / "meta.tar.gz"
    tar_path.write_bytes(buffer.getvalue())
    dest = tmp_path / "out"
    dest.mkdir()
    listing, _ = service._extract_tarball(tar_path, dest)
    assert listing == ["real.py"]
    assert not (dest / ".albedo-listing.json").exists()


def test_cache_limit_clears_snapshots_dir(tmp_path, monkeypatch):
    service = make_service(tmp_path, max_cache_gb=1024 / 1024**3)
    old_snapshot = make_snapshot(service, {"src/app.py": "x"})
    marker = old_snapshot / ".albedo-repo-context-done"
    marker.write_text(json.dumps({"bytes": 2048}))

    def fake_download(owner, repo, sha, dest):
        dest.write_bytes(_tar_bytes())

    monkeypatch.setattr(service, "_download_tarball", fake_download)
    new_sha = "0" * 40
    fresh = service._ensure_snapshot("other", "repo", new_sha)
    assert fresh is not None and (fresh / "src/app.py").exists()
    assert not old_snapshot.exists()

    roomy = make_service(tmp_path, cache_dir=str(tmp_path / "cache2"), max_cache_gb=60.0)
    kept = make_snapshot(roomy, {"src/app.py": "x"})
    monkeypatch.setattr(roomy, "_download_tarball", fake_download)
    assert roomy._ensure_snapshot("other", "repo", new_sha) is not None
    assert kept.exists()


def test_prefetch_dedupes_and_skips_already_downloaded(tmp_path, monkeypatch):
    manifest = {
        "sources": [
            {
                "name": "swe-zero",
                "shards": [
                    {
                        "path": "swe-zero/data/train-00000.parquet",
                        "rows": 3,
                        "rows_meta": [{"iid": "o__r-1"}, {"iid": "o__r-1"}, {"iid": "o__r-2"}],
                    }
                ],
            }
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    service = make_service(tmp_path, dataset_manifest_path=str(manifest_path))
    make_snapshot(service, {"src/app.py": "x"})

    new_sha = "1" * 40
    monkeypatch.setattr(
        service,
        "_resolve_sha",
        lambda ref: ("o", "r", FULL_SHA if ref.instance_id == "o__r-2" else new_sha),
    )
    downloads = []

    def fake_download(owner, repo, sha, dest):
        downloads.append(sha)
        dest.write_bytes(_tar_bytes())

    monkeypatch.setattr(service, "_download_tarball", fake_download)
    summary = service.prefetch(
        [
            "swe-zero/data/train-00000.parquet:0:0",
            "swe-zero/data/train-00000.parquet:1:2",
            "swe-zero/data/train-00000.parquet:2:0",
            "garbage",
        ]
    )
    assert summary == {"samples": 4, "instances": 2, "ready": 2}
    assert downloads == [new_sha]

    again = service.prefetch(
        ["swe-zero/data/train-00000.parquet:0:0", "swe-zero/data/train-00000.parquet:2:0"]
    )
    assert again["ready"] == 2
    assert downloads == [new_sha]


def test_oversized_snapshot_failure_is_cached(tmp_path, monkeypatch):
    service = make_service(tmp_path)
    calls = []

    def too_large(owner, repo, sha, dest):
        calls.append(sha)
        raise _SnapshotTooLarge("tarball exceeds limit")

    monkeypatch.setattr(service, "_download_tarball", too_large)
    assert service._ensure_snapshot("o", "r", FULL_SHA) is None
    assert service._ensure_snapshot("o", "r", FULL_SHA) is None
    assert len(calls) == 1


def test_a_chain_still_grounds_the_stages_after_one_it_cannot_render(tmp_path, monkeypatch):
    service = make_service(tmp_path)
    make_snapshot(service, {"a.py": "L1\nL2\nL3\n"})
    monkeypatch.setattr(service, "_resolve_sha", lambda ref: ("o", "r", FULL_SHA))

    # a git stage first: the stages after it are still grounded
    after_git = service.repo_context_for_instance(
        "swe-zero", "o__r-1", "```bash\ngit show HEAD --stat && sed -n '1,2p' a.py\n```"
    )
    assert "CHAIN STAGES —" in after_git.context
    assert "$ sed -n '1,2p' a.py\nL1\nL2" in after_git.context
    assert after_git.exact_output is None

    # and a stage between two ungroundable ones is still reported
    sandwiched = service.repo_context_for_instance(
        "swe-zero", "o__r-1", "```bash\npwd && sed -n '3p' a.py && python run.py\n```"
    )
    assert "$ sed -n '3p' a.py\nL3" in sandwiched.context


def test_a_ranged_read_accepts_one_line_and_a_pipe_stage():
    listing = ["m.py"]
    src = "def a():\n    x = 1\n    return x\n\ndef b():\n    return 2\n"
    read = {"m.py": src}.get

    def out(cmd):
        return run_search(parse_search(cmd), read, listing).output

    assert out("sed -n '3p' m.py") == "    return x"
    assert out("sed -n '2,3p' m.py") == "    x = 1\n    return x"
    assert out("nl -ba m.py | sed -n '2,3p'") == "     2\t    x = 1\n     3\t    return x"
    assert out("cat m.py | sed -n '1,2p'") == "def a():\n    x = 1"
    # only numeric addresses are modelled: three corpus commands use a regex range,
    # so resolving pattern addresses is not worth the machinery
    assert isinstance(parse_search("sed -n '/def a/,/return x/p' m.py"), ParseFailure)
    assert isinstance(parse_search("cat m.py | sed -n '/a/,/b/p'"), ParseFailure)
    assert isinstance(parse_search("sed -n '2,3d' m.py"), ParseFailure)


def test_a_search_over_an_unreadable_file_is_never_reported_as_no_match(tmp_path, monkeypatch):
    service = make_service(tmp_path)
    make_snapshot(service, {"a.py": "    def target():\n        pass\n"})
    monkeypatch.setattr(service, "_resolve_sha", lambda ref: ("o", "r", FULL_SHA))

    # a sed -i form we deliberately refuse to model, so the path is left dirty
    edited = [
        {"role": "assistant", "content": "```bash\nsed -i '/pass/d' a.py\n```"},
        {"role": "user", "content": "<returncode>0</returncode>\n<output>\n</output>"},
    ]
    dirty = service.repo_context_for_instance(
        "swe-zero", "o__r-1", "```bash\ngrep -n target a.py\n```", edited
    )
    assert "matched NOTHING" not in dirty.context
    assert dirty.exact_output is None

    clean = service.repo_context_for_instance(
        "swe-zero", "o__r-1", "```bash\ngrep -n target a.py\n```"
    )
    assert clean.exact_output == "1:    def target():"


def test_posix_classes_and_stderr_redirects_do_not_silence_a_grep():
    listing = ["m.py"]
    read = {"m.py": "class A:\n    def go(self):\n        pass\n"}.get
    hit = run_search(parse_search("grep -n '[[:space:]]*def' m.py"), read, listing)
    assert hit.output == "2:    def go(self):"
    assert not hit.empty
    # 2>&1 is a redirect, not a control operator
    assert not isinstance(parse_search("grep -rn target m.py 2>&1"), ParseFailure)
    assert not isinstance(parse_search("grep -rn target m.py 2>&1 | head -20"), ParseFailure)
    # sending stdout elsewhere is still not something we can claim was printed
    assert isinstance(parse_search("grep -n target m.py 1>&2"), ParseFailure)


def test_computed_search_reports_the_exit_status_the_shell_would(tmp_path, monkeypatch):
    service = make_service(tmp_path)
    make_snapshot(service, {"src/app.py": "needle here\n", "src/util.py": "nothing\n"})
    monkeypatch.setattr(service, "_resolve_sha", lambda ref: ("o", "r", FULL_SHA))
    run = lambda cmd: service.repo_context_for_instance(  # noqa: E731
        "swe-zero", "o__r-1", f"```bash\n{cmd}\n```"
    )

    assert run("grep -rn needle src/").exact_returncode == 0
    # grep found nothing, which a shell reports as failure rather than an empty success
    assert run("grep -rn absent-token src/").exact_returncode == 1
    assert run("cat src/app.py").exact_returncode == 0
    assert run("cat src/nope.py").exact_returncode == 1
    # find reports success when nothing matched
    assert run("find src -name '*.rs'").exact_returncode == 0
    # the last command in a pipeline owns the status, and wc always succeeds
    assert run("grep -rn absent-token src/ | wc -l").exact_returncode == 0


def test_no_exit_status_is_claimed_for_a_search_that_cannot_be_stood_behind(tmp_path, monkeypatch):
    service = make_service(tmp_path)
    make_snapshot(service, {"src/app.py": "needle here\n"})
    monkeypatch.setattr(service, "_resolve_sha", lambda ref: ("o", "r", FULL_SHA))

    # verified against GNU grep: `-c` prints a "0" line and still fails, as nothing matched
    counted = service.repo_context_for_instance(
        "swe-zero", "o__r-1", "```bash\ngrep -rc absent-token src/\n```"
    )
    assert (counted.exact_output, counted.exact_returncode) == ("src/app.py:0", 1)
    # a status this cannot stand behind is never claimed: the search is left to the simulator
    unknown = service.repo_context_for_instance(
        "swe-zero", "o__r-1", "```bash\npython run.py | grep -c needle\n```"
    )
    assert unknown.exact_returncode is None


def _search(cmd: str, files: dict[str, str]):
    plan = parse_search(cmd)
    assert not isinstance(plan, ParseFailure), f"{cmd}: {plan}"
    return run_search(plan, lambda rel: files.get(rel), sorted(files))


_GREP_FIXTURE = {"b.py": "alpha\nneedle one\nbeta\ngamma\ndelta\nneedle two\nomega\n"}


def test_a_piped_grep_reproduces_context_flags_the_way_grep_prints_them():
    # verified against GNU grep: a lone `--` between groups, `N:` on a hit, `N-` on context
    assert _search("cat b.py | grep -A 1 needle", _GREP_FIXTURE).output == (
        "needle one\nbeta\n--\nneedle two\nomega"
    )
    assert _search("cat b.py | grep -A1 -B1 needle", _GREP_FIXTURE).output == (
        "alpha\nneedle one\nbeta\n--\ndelta\nneedle two\nomega"
    )
    assert _search("cat b.py | grep -C 1 needle", _GREP_FIXTURE).output == (
        "alpha\nneedle one\nbeta\n--\ndelta\nneedle two\nomega"
    )
    assert _search("cat b.py | grep -n -A 1 needle", _GREP_FIXTURE).output == (
        "2:needle one\n3-beta\n--\n6:needle two\n7-omega"
    )
    # a context count that is not a number, and -v with context, are not modelled
    assert isinstance(parse_search("cat b.py | grep -A absent needle"), ParseFailure)
    assert isinstance(parse_search("cat b.py | grep -v -A 1 needle"), ParseFailure)


def test_an_awk_line_window_reads_like_the_sed_range_it_is():
    files = {"a.py": "".join(f"line {i}\n" for i in range(1, 21))}
    assert _search("awk 'NR>=3 && NR<=5' a.py", files).output == "line 3\nline 4\nline 5"
    assert _search("awk 'NR >= 3 && NR <= 5' a.py", files).output == "line 3\nline 4\nline 5"
    # an awk program that also prints its own formatting is not a plain window
    assert isinstance(parse_search("awk 'NR>=3 && NR<=5 {print NR}' a.py"), ParseFailure)
    assert isinstance(parse_search("awk '/def x/,/return/' a.py"), ParseFailure)


def test_wc_l_counts_the_lines_and_names_the_operand():
    files = {"a.py": "".join(f"line {i}\n" for i in range(1, 21))}
    assert _search("wc -l a.py", files).output == "20 a.py"
    # verified against GNU wc: counts pad to the width of the files' total size
    both = {**files, "b.py": "x\n"}
    assert _search("wc -l a.py b.py", both).output == " 20 a.py\n  1 b.py\n 21 total"
    assert _search("wc a.py", files).output == " 20  40 151 a.py"
    assert _search("cat a.py | wc", files).output == "     20      40     151"
    # reading the terminal is not modelled
    assert isinstance(parse_search("wc -l"), ParseFailure)


def test_a_read_of_a_missing_file_exits_the_way_its_own_tool_does():
    files = {"a.py": "one\ntwo\n"}
    # verified against GNU coreutils: sed exits 2 when it cannot open its input, cat/head/nl 1
    assert _search("sed -n '1,2p' gone.py", files).returncode == 2
    for cmd in ("cat gone.py", "head -5 gone.py", "nl -ba gone.py", "wc -l gone.py"):
        assert _search(cmd, files).returncode == 1, cmd


def test_cat_numbers_several_operands_as_one_stream():
    listing = ["a.py", "b.py"]
    read = {"a.py": "one\ntwo\n", "b.py": "three\n"}.get
    assert run_search(parse_search("cat a.py b.py"), read, listing).output == "one\ntwo\nthree"
    numbered = run_search(parse_search("cat -n a.py b.py"), read, listing).output
    assert numbered == "     1\tone\n     2\ttwo\n     3\tthree"
    # head and tail banner each operand
    banners = run_search(parse_search("head -1 a.py b.py"), read, listing).output
    assert banners == "==> a.py <==\none\n\n==> b.py <==\nthree"
    # a file without its final line break runs into the next one, as cat writes it
    joined = run_search(
        parse_search("cat a.py b.py"), {"a.py": "one", "b.py": "two\n"}.get, listing
    )
    assert joined.output == "onetwo"


def test_a_heredoc_body_is_never_reported_as_a_missing_file():
    listing = ["src/docx/text/run.py"]
    write = "cat > /tmp/repro.py <<'EOF'\nimport datetime as dt\nprint(dt.datetime.now())\nEOF"
    present, missing = _referenced_paths(write, listing)
    assert missing == []
    # a real repository path named inside the body is still worth reporting as present
    reads = "cat > /tmp/fix.py <<'EOF'\ncontent = open('src/docx/text/run.py').read()\nEOF"
    assert _referenced_paths(reads, listing) == (["src/docx/text/run.py"], [])
    # and a path the command itself names is still asserted absent
    assert _referenced_paths("sed -i 's/a/b/' nope/gone.py", listing)[1] == ["nope/gone.py"]


def test_a_sed_script_is_never_reported_as_a_missing_file():
    listing = ["src/docx/text/run.py"]
    edit = "sed -i '112s/self.font.bold = True/self.font.bold = None/' src/docx/text/run.py"
    assert _referenced_paths(edit, listing)[1] == []
    assert _data_words("sed -i -e 's/a/b/' -e 's/c/d/' src/x.py") == ["s/a/b/", "s/c/d/"]
    # the file the script runs against is still checked
    assert _referenced_paths("sed -i 's/a/b/' nope/gone.py", listing)[1] == ["nope/gone.py"]


def test_inline_code_and_patterns_are_never_reported_as_missing_files():
    """A module path in `python -c` code, or a dotted name grep searches for, is not a file: told
    it is absent, the simulator answers `ModuleNotFoundError` for an import that works."""
    listing = ["src/docx/text/run.py", "README.md"]
    for command in (
        "python3 -c 'from src.docx.text import run; print(run)'",
        "grep -rn src.docx.text src",
        "awk '/src.docx.x/' README.md",
    ):
        assert _referenced_paths(command, listing)[1] == [], command
    # a file the code opens is still shown
    assert _referenced_paths("python3 -c \"open('README.md')\"", listing)[0] == ["README.md"]


def test_an_installed_package_is_never_reported_missing_but_a_checkout_file_is():
    """A file outside the checkout (an installed package) is not the repo's to know: told it is
    absent, the simulator answers `No such file` for a file that exists. The installed copy of
    the project itself ends with a checkout path and must not be taken for it either."""
    listing = ["src/docx/text/run.py", "README.md"]
    for command in (
        "cat /usr/lib/python3.11/site-packages/docx/text/run.py",
        "ls /opt/conda/lib/python3.9/site-packages/docx",
    ):
        assert _referenced_paths(command, listing) == ([], []), command
    for command in ("cat /testbed/src/docx/gone.py", "cat /workspace/o__r__1.0/src/docx/gone.py"):
        assert _referenced_paths(command, listing)[1] == ["src/docx/gone.py"], command


def _chain_result(service, command: str):
    return service.repo_context_for_instance("swe-zero", "o__r-12", f"```bash\n{command}\n```")


def test_a_chain_of_repository_queries_is_answered_as_one_command(tmp_path, monkeypatch):
    service = make_service(tmp_path)
    make_snapshot(service, {"a.py": "alpha\n", "b.py": "beta\n"})
    monkeypatch.setattr(service, "_resolve_sha", lambda ref: ("o", "r", FULL_SHA))

    result = _chain_result(service, "cat a.py && cat b.py")
    assert result.exact_output == "alpha\nbeta"
    assert result.exact_returncode == 0


def test_a_chain_stops_at_the_stage_that_fails_and_reports_its_status(tmp_path, monkeypatch):
    service = make_service(tmp_path)
    make_snapshot(service, {"a.py": "alpha\n", "b.py": "beta\n"})
    monkeypatch.setattr(service, "_resolve_sha", lambda ref: ("o", "r", FULL_SHA))

    # grep exits 1 when nothing matched, so the shell never runs the second read and the
    # observation must not contain its output
    result = _chain_result(service, "grep nowhere a.py && cat b.py")
    assert result.exact_returncode == 1
    assert result.exact_output == ""
    assert "beta" not in (result.context or "")


def test_a_chain_applies_each_edit_before_the_stages_after_it(tmp_path, monkeypatch):
    """A read after an edit in the same command sees the edit, as it would in a shell."""
    service = make_service(tmp_path)
    make_snapshot(service, {"a.py": "alpha\n"})
    monkeypatch.setattr(service, "_resolve_sha", lambda ref: ("o", "r", FULL_SHA))

    edited = _chain_result(service, "sed -i 's/alpha/omega/' a.py && cat a.py")
    assert (edited.exact_output, edited.exact_returncode) == ("omega", 0)
    created = _chain_result(service, "cat > new.py <<'EOF'\nbody\nEOF\ncat new.py")
    assert created.exact_output == "body"
    lone = _chain_result(service, "sed -i 's/alpha/omega/' a.py")
    assert (lone.exact_output, lone.exact_returncode) == ("", 0)
    # the evidence for a stage that cannot run is taken after the edit, never before it
    mixed = _chain_result(service, "sed -i 's/alpha/omega/' a.py && cat a.py && python run.py")
    assert mixed.exact_output is None
    assert "$ cat a.py\nomega" in mixed.context
    # and the file contents the simulator gets are the edited ones
    assert "--- ./a.py ---\nomega" in mixed.context
    assert "\nalpha\n" not in mixed.context


def test_a_chain_answers_its_reads_after_the_writes_before_them(tmp_path, monkeypatch):
    service = make_service(tmp_path)
    make_snapshot(service, {"a.py": "alpha\n"})
    monkeypatch.setattr(service, "_resolve_sha", lambda ref: ("o", "r", FULL_SHA))

    copied = _chain_result(service, "cp a.py b.py && cat b.py")
    assert (copied.exact_output, copied.exact_returncode) == ("alpha", 0)
    redirected = _chain_result(service, "cat a.py > copy.txt && cat copy.txt")
    assert redirected.exact_output == "alpha"
    # a listing after a removal must not show the removed file
    removed = _chain_result(service, "rm a.py && ls .")
    assert (removed.exact_output, removed.exact_returncode) == ("", 0)
    moved = _chain_result(service, "mv a.py b.py && cat a.py")
    assert moved.exact_output == "cat: a.py: No such file or directory"
    assert moved.exact_returncode == 1
    # a program's writes are not known: what it may have written is not answered
    unknown = _chain_result(service, "python gen.py > a.py && cat a.py")
    assert unknown.exact_output is None


def test_a_canned_stage_answers_for_itself_inside_a_chain(tmp_path, monkeypatch):
    service = make_service(tmp_path)
    make_snapshot(service, {"a.py": "alpha\n", "b.py": "beta\n"})
    monkeypatch.setattr(service, "_resolve_sha", lambda ref: ("o", "r", FULL_SHA))

    after = _chain_result(service, "cat a.py && pytest -q; cat b.py")
    assert (after.exact_output, after.exact_returncode) == (f"alpha\n{PYTEST_MISSING}\nbeta", 0)
    stopped = _chain_result(service, "cat a.py && pytest -q && cat b.py")
    assert (stopped.exact_output, stopped.exact_returncode) == (f"alpha\n{PYTEST_MISSING}", 1)


def test_a_cd_moves_the_stages_after_it(tmp_path, monkeypatch):
    service = make_service(tmp_path)
    make_snapshot(service, {"a.py": "alpha\n", "pkg/a.py": "different\n"})
    monkeypatch.setattr(service, "_resolve_sha", lambda ref: ("o", "r", FULL_SHA))

    # a read after `cd pkg` names pkg/a.py, which a search from the root would miss
    moved = _chain_result(service, "cat a.py && cd pkg && cat a.py")
    assert moved.exact_output is None
    assert "$ cat a.py\nalpha" in moved.context
    assert _chain_result(service, "cd /testbed && cat a.py && cat a.py").exact_output == (
        "alpha\nalpha"
    )
    # a cd that fails stops the `&&` chain, with bash's own message
    failed = _chain_result(service, "cd missing && cat a.py")
    assert failed.exact_output == "bash: cd: missing: No such file or directory"
    assert failed.exact_returncode == 1


def test_mini_swe_agent_commands_fail_in_dash_words(tmp_path, monkeypatch):
    service = make_service(tmp_path)
    make_snapshot(service, {"a.py": "alpha\n"})
    monkeypatch.setattr(service, "_resolve_sha", lambda ref: ("o", "r", FULL_SHA))

    failed = service.repo_context_for_instance(
        "open-swe-traces-v1.2", "o__r-12", "```bash\ncd missing && cat a.py\n```", fmt="returncode"
    )
    assert failed.exact_output == "/bin/sh: 1: cd: can't cd to missing"
    assert failed.exact_returncode == 2


def test_a_missing_path_is_reported_unless_an_observation_showed_it(tmp_path, monkeypatch):
    service = make_service(tmp_path)
    make_snapshot(service, {"a.py": "alpha\n"})
    monkeypatch.setattr(service, "_resolve_sha", lambda ref: ("o", "r", FULL_SHA))

    missing = _chain_result(service, "cat a.py && grep -rn x gone/")
    assert missing.exact_output == "alpha\ngrep: gone/: No such file or directory"
    assert missing.exact_returncode == 2
    # a path an earlier observation listed exists on the machine, tracked or not
    listing = (
        "Here's the files and directories up to 2 levels deep in /testbed, excluding hidden"
        " items:\n/testbed/\n/testbed/a.py\n/testbed/gone/\n/testbed/gone/x.txt\n"
    )
    shown = [
        {"role": "assistant", "content": "```bash\nls /testbed\n```"},
        {"role": "user", "content": listing},
    ]
    attested = service.repo_context_for_instance(
        "swe-zero", "o__r-12", "```bash\ncat a.py && grep -rn x gone/\n```", shown
    )
    assert attested.exact_output is None


def test_pwd_answers_from_the_session_root_and_declines_without_one():
    plan = parse_search("pwd")
    assert not isinstance(plan, ParseFailure)
    assert run_search(plan, lambda rel: None, [], root="/workspace/o__r__1.0").output == (
        "/workspace/o__r__1.0"
    )
    # no root anywhere in the transcript means no directory to print
    assert isinstance(run_search(parse_search("pwd"), lambda rel: None, [], root=""), ParseFailure)
    # after a `cd`, pwd prints where it went, not the session root: that is not answered here
    assert isinstance(parse_search("cd /elsewhere && pwd"), ParseFailure)
    assert isinstance(parse_search("pwd -P"), ParseFailure)


HEAD_SHA = "1" * 40
BUG_SHA = "2" * 40
INITIAL_SHA = "3" * 40
SMITH_ID = "BurntSushi__ripgrep.3b7fd442.func_pm_flip_operators__3m5orim"


def _smith_head(*subjects: str) -> dict:
    """The commits/<branch> payload for a swesmith branch whose history is `subjects`, newest
    first."""
    shas = [HEAD_SHA, BUG_SHA, INITIAL_SHA]
    return {
        "sha": shas[0],
        "commit": {"message": subjects[0]},
        "parents": [{"sha": shas[1]}] if len(subjects) > 1 else [],
    }


def test_smith_mirror_points_mini_coder_ids_at_the_swesmith_branch():
    ref = parse_instance("mini-coder", SMITH_ID)
    mirror = _smith_mirror(ref)
    assert (mirror.owner, mirror.repo, mirror.commit) == (
        "swesmith",
        "BurntSushi__ripgrep.3b7fd442",
        SMITH_ID,
    )
    assert mirror.instance_id == ref.instance_id  # the sha cache stays keyed by instance id
    assert _smith_mirror(parse_instance("swe-hero", f"pandas-dev__pandas-{FULL_SHA}")) is None
    assert _smith_mirror(parse_instance("open-swe-traces", "python-attrs__attrs-770")) is None


@pytest.mark.parametrize(
    ("source", "history", "expected"),
    [
        # Python/Go trajectories ran at the branch head: bug in place, F2P test files deleted.
        ("mini-coder", ("Remove F2P Tests", "Bug Patch", "Initial commit"), HEAD_SHA),
        ("mini-coder", ("Bug Patch", "Initial commit"), HEAD_SHA),
        # The Rust harness ran at Bug Patch; there the head deletes whole source files.
        ("mini-coder-rs", ("Remove F2P Tests", "Bug Patch", "Initial commit"), BUG_SHA),
        # ...unless the branch never got a removal commit, so the head is Bug Patch itself.
        ("mini-coder-rs", ("Bug Patch", "Initial commit"), HEAD_SHA),
    ],
)
def test_resolve_sha_smith_mirror_picks_the_commit_the_agent_ran_on(
    tmp_path, monkeypatch, source, history, expected
):
    service = make_service(tmp_path)
    calls = []

    def fake_github_json(path):
        calls.append(path)
        assert path == f"/repos/swesmith/BurntSushi__ripgrep.3b7fd442/commits/{SMITH_ID}"
        return _smith_head(*history)

    monkeypatch.setattr(service, "_github_json", fake_github_json)
    ref = parse_instance(source, SMITH_ID)
    assert service._resolve_sha(ref) == ("swesmith", "BurntSushi__ripgrep.3b7fd442", expected)
    assert len(calls) == 1


def test_resolve_sha_swe_hero_uses_the_parent_of_the_fix_commit(tmp_path, monkeypatch):
    """R2E-Gym ids name the fix commit; the trajectory was recorded on its parent."""
    service = make_service(tmp_path)
    fix, parent = "4a" * 20, "5b" * 20  # an all-digit tail would parse as a PR number

    def fake_github_json(path):
        assert path == f"/repos/pandas-dev/pandas/commits/{fix}"
        return {"sha": fix, "commit": {"message": "BUG: x (#1)"}, "parents": [{"sha": parent}]}

    monkeypatch.setattr(service, "_github_json", fake_github_json)
    ref = parse_instance("swe-hero", f"pandas-dev__pandas-{fix}")
    assert service._resolve_sha(ref) == ("pandas-dev", "pandas", parent)


def test_resolve_sha_affine_r2e_ids_use_the_parent_of_the_fix_commit(tmp_path, monkeypatch):
    service = make_service(tmp_path)
    fix, parent = "4a" * 20, "5b" * 20
    monkeypatch.setattr(
        service,
        "_github_json",
        lambda path: {"sha": fix, "commit": {"message": "x"}, "parents": [{"sha": parent}]},
    )
    ref = parse_instance("affine-mswea", f"pandas-dev__pandas-{fix}")
    assert service._resolve_sha(ref) == ("pandas-dev", "pandas", parent)


def test_resolve_sha_affine_smith_ids_are_cached_apart_from_mini_coder_rs(tmp_path, monkeypatch):
    """Affine's Rust machines ran at the branch head, mini-coder-rs's at Bug Patch: one id, two
    trees."""
    service = make_service(tmp_path)
    monkeypatch.setattr(
        service,
        "_github_json",
        lambda path: _smith_head("Remove F2P Tests", "Bug Patch", "Initial commit"),
    )
    mirror = "BurntSushi__ripgrep.3b7fd442"
    assert service._resolve_sha(parse_instance("mini-coder-rs", SMITH_ID)) == (
        "swesmith",
        mirror,
        BUG_SHA,
    )
    assert service._resolve_sha(parse_instance("mini-coder-affine-tools", SMITH_ID)) == (
        "swesmith",
        mirror,
        HEAD_SHA,
    )
    assert service._resolve_sha(parse_instance("mini-coder-rs", SMITH_ID)) == (
        "swesmith",
        mirror,
        BUG_SHA,
    )


def test_resolve_sha_plain_commit_ids_keep_using_the_commit_itself(tmp_path, monkeypatch):
    service = make_service(tmp_path)

    def fake_github_json(path):
        assert path == f"/repos/own/repo/commits/{FULL_SHA}"
        return {"sha": FULL_SHA, "commit": {"message": "x"}, "parents": [{"sha": "6" * 40}]}

    monkeypatch.setattr(service, "_github_json", fake_github_json)
    ref = parse_instance("open-swe-traces", f"own__repo-{FULL_SHA}")
    assert service._resolve_sha(ref) == ("own", "repo", FULL_SHA)


def test_resolve_sha_ignores_cache_entries_written_under_an_older_rule(tmp_path, monkeypatch):
    """Entries from before the swesmith/swe-hero rules point at the clean upstream tree. They
    must be resolved again rather than served, so a deploy needs no manual cache purge."""
    service = make_service(tmp_path)
    cache_file = service.cache_dir / "shas" / f"{_safe_name(SMITH_ID)}.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps({"owner": "BurntSushi", "repo": "ripgrep", "sha": FULL_SHA}))
    calls = []

    def fake_github_json(path):
        calls.append(path)
        return _smith_head("Remove F2P Tests", "Bug Patch", "Initial commit")

    monkeypatch.setattr(service, "_github_json", fake_github_json)
    ref = parse_instance("mini-coder", SMITH_ID)
    assert service._resolve_sha(ref) == ("swesmith", "BurntSushi__ripgrep.3b7fd442", HEAD_SHA)
    assert json.loads(cache_file.read_text())["rule"] == _SHA_RULE
    assert service._resolve_sha(ref) == ("swesmith", "BurntSushi__ripgrep.3b7fd442", HEAD_SHA)
    assert len(calls) == 1


def test_a_missing_path_prints_nothing_when_stderr_is_discarded():
    files = {"a.py": "x\n"}
    for command, code in (
        ("cat gone.py 2>/dev/null", 1),
        ("ls gone/ 2>/dev/null", 2),
        ("grep -rn x gone/ 2>/dev/null", 2),
    ):
        loud = _search(command.replace(" 2>/dev/null", ""), files)
        quiet = _search(command, files)
        assert "No such file or directory" in loud.output, command
        assert quiet.output == "" and quiet.missing, command
        assert quiet.returncode == loud.returncode == code, command


def test_a_gap_is_simulated_with_the_modules_its_program_imports(tmp_path, monkeypatch):
    """`python3 run.py` prints what the modules it imports compute: the gap's context shows
    them, as the earlier stages of the same command left them."""
    service = make_service(tmp_path)
    make_snapshot(
        service,
        {
            "run.py": "from pkg import app\nprint(app.SIZE)\n",
            "pkg/__init__.py": "",
            "pkg/app.py": "SIZE = 1\n",
        },
    )
    monkeypatch.setattr(service, "_resolve_sha", lambda ref: ("o", "r", FULL_SHA))

    result = _chain_result(
        service, "sed -i 's/SIZE = 1/SIZE = 5/' pkg/app.py && cat run.py && python3 run.py"
    )
    gap = next(part for part in result.parts if part["kind"] == "gap")
    assert "--- ./pkg/app.py ---\nSIZE = 5" in gap["context"]
    assert "--- ./pkg/app.py ---\nSIZE = 1" not in gap["context"]


MB_FILE = ("x" * 99 + "\n") * 10486  # just over 1 MB


def _session(service, commands: list[str], command: str):
    messages = []
    for earlier in commands:
        messages += [
            {"role": "assistant", "content": f"```bash\n{earlier}\n```"},
            {"role": "user", "content": "<returncode>0</returncode>\n<output>\n</output>"},
        ]
    return service.repo_context_for_instance(
        "swe-zero", "o__r-12", f"```bash\n{command}\n```", messages, fmt="returncode"
    )


def test_a_command_asking_for_more_text_than_is_answered_is_left_to_the_simulator(
    tmp_path, monkeypatch
):
    """A few words can ask for gigabytes - a brace sequence, one file read many times, a file
    doubled by each turn of a replayed session, a replacement repeated at every character.
    None of it is built: the command is left to the simulator, and ordinary uses are answered."""
    service = make_service(tmp_path)
    make_snapshot(service, {"big.txt": MB_FILE, "f1": "1\n", "f2": "2\n", "a.txt": "aaaa\n"})
    monkeypatch.setattr(service, "_resolve_sha", lambda ref: ("o", "r", FULL_SHA))

    assert _session(service, [], "echo {1..3000000}").exact_output is None
    assert (
        _session(service, [], "cat f{1..2} && echo {a,b}{1,2}").exact_output == "1\n2\na1 a2 b1 b2"
    )

    assert _session(service, [], "cat " + " big.txt" * 20 + " | wc -l").exact_output is None
    assert _session(service, [], "cat big.txt big.txt | wc -l").exact_output == "20972"

    doubling = ["cp big.txt c.txt"] + ["cat c.txt c.txt > d.txt && mv d.txt c.txt"] * 8
    assert _session(service, doubling, "wc -c c.txt").exact_output is None
    assert _session(service, doubling[:3], "wc -l c.txt").exact_output == "41944 c.txt"

    widened = f"sed -i 's/x/{'y' * 1000}/g' big.txt"
    assert _session(service, [widened], "wc -c big.txt").exact_output is None
    assert _session(service, ["sed -i 's/a/bb/g' a.txt"], "cat a.txt").exact_output == "bbbbbbbb"


def test_an_editor_view_is_the_openhands_editor_s_rendering_of_the_file():
    """In datasets whose `cat -n` and `sed -n 'A,Bp' f | cat -n` were converted from editor
    views, the recorded observation is the editor's: tabs kept, every piece between line breaks
    numbered (from A for a range), the text clipped at the editor's limit. A range the editor
    refuses is an editor error, which is left to the simulator."""
    text = "def f():\n\treturn 1\n\nx = 2\n"
    assert (
        _editor_view(text)
        == "     1\tdef f():\n     2\t\treturn 1\n     3\t\n     4\tx = 2\n     5"
    )
    assert _editor_view(text, 2, 3) == "     2\t\treturn 1\n     3"
    assert _editor_view(text, 4, 9) is None and _editor_view(text, 3, 2) is None
    clipped = _editor_view("a" * 20000)
    assert clipped.startswith("     1\t" + "a" * 100) and clipped.endswith("looking for.</NOTE>")
    assert len(clipped) == len("     1\t") + 16000 + len(clipped.split("a" * 16000)[1])
    served = grounded_observation(OPENHANDS, "     2\tx", 0, "sed -n '2,2p' /w/f.py | cat -n", [])
    assert served == "Here's the result of running `cat -n` on /w/f.py:\n     2\tx"


def test_an_editor_read_of_a_directory_leaves_no_bash_answer_for_the_simulator(
    tmp_path, monkeypatch
):
    """The editor lists a directory in the filesystem's order, which is not known: the read is
    simulated, and bash's `Is a directory` must not reach the simulator as a computed answer."""
    service = make_service(tmp_path)
    make_snapshot(service, {"pkg/a.py": "x = 1\n"})
    monkeypatch.setattr(service, "_resolve_sha", lambda ref: ("o", "r", FULL_SHA))
    monkeypatch.setattr(
        core, "SCAFFOLDS", {("swe-zero", "openhands"): core.Scaffold(editor=core._EDITOR_CLIPPED)}
    )
    for command, exact in (
        ("cat -n /testbed/pkg", None),
        ("cat -n /testbed/pkg/a.py", "     1\tx = 1\n     2"),
    ):
        result = service.repo_context_for_instance(
            "swe-zero", "o__r-12", f"```bash\n{command}\n```", fmt="openhands"
        )
        assert result.exact_output == exact, command
        assert "Is a directory" not in (result.context or "") and not result.parts


def _format_patch(sha: str, subject: str, diff: list[str]) -> str:
    return "\n".join(
        [
            f"From {sha} Mon Sep 17 00:00:00 2001",
            "From: swesmith <swesmith@swesmith.ai>",
            "Date: Sat, 12 Jul 2025 17:08:47 +0000",
            f"Subject: [PATCH] {subject}",
            "",
            "---",
            " a.py | 2 +-",
            " 1 file changed, 1 insertion(+), 1 deletion(-)",
            "",
            *diff,
            "",
        ]
    )


_BUG_DIFF = [
    "diff --git a/a.py b/a.py",
    "index 1111111..2222222 100644",
    "--- a/a.py",
    "+++ b/a.py",
    "@@ -1,2 +1,2 @@",
    " def add(a, b):",
    "-    return a + b",
    "+    return a - b",
]
_F2P_DIFF = [
    "diff --git a/tests/test_a.py b/tests/test_a.py",
    "deleted file mode 100644",
    "index 3333333..0000000",
    "--- a/tests/test_a.py",
    "+++ /dev/null",
    "@@ -1,2 +0,0 @@",
    "-def test_add():",
    "-    assert add(1, 2) == 3",
]


def _fake_github(
    monkeypatch,
    service,
    history: list[tuple[str, str]],
    patches: dict[str, str],
    upstream: dict[str, dict] | None = None,
):
    calls = []

    def fake_json(path):
        calls.append(path)
        if path in (upstream or {}):
            return upstream[path]
        if "/commits?" in path:
            return [{"sha": sha, "commit": {"message": subject}} for sha, subject in history]
        if "/commits/" in path:
            raise _NotFound(path)
        return {"default_branch": "main"}

    def fake_text(path, accept):
        calls.append(path)
        rev = path.rsplit("/", 1)[-1]
        return next((text for sha, text in patches.items() if sha.startswith(rev)), "")

    monkeypatch.setattr(service, "_github_json", fake_json)
    monkeypatch.setattr(service, "_github_text", fake_text)
    return calls


def test_smith_mirror_history_is_served_as_one_squashed_commit(tmp_path, monkeypatch):
    service = make_service(tmp_path)
    snapshot = make_snapshot(service, {"a.py": "def add(a, b):\n    return a - b\n"})
    mirror = "BurntSushi__ripgrep.3b7fd442"
    monkeypatch.setattr(service, "_resolve_sha", lambda ref: ("swesmith", mirror, HEAD_SHA))
    monkeypatch.setattr(service, "_ensure_snapshot", lambda owner, repo, sha: snapshot)
    calls = _fake_github(
        monkeypatch,
        service,
        [(HEAD_SHA, "Remove F2P Tests"), (BUG_SHA, "Bug Patch"), (INITIAL_SHA, "Initial commit")],
        {
            HEAD_SHA: _format_patch(HEAD_SHA, "Remove F2P Tests", _F2P_DIFF),
            BUG_SHA: _format_patch(BUG_SHA, "Bug Patch", _BUG_DIFF),
        },
    )

    def ground(command):
        return service.repo_context_for_instance("mini-coder", SMITH_ID, f"```bash\n{command}\n```")

    squashed = f"{HEAD_SHA[:7]} Initial commit"
    assert ground("git log --oneline -5").exact_output == squashed
    assert ground("git log --oneline -- a.py").exact_output == squashed
    upstream = "https://github.com/BurntSushi/ripgrep"
    assert ground("git remote -v").exact_output == (
        f"origin\t{upstream} (fetch)\norigin\t{upstream} (push)"
    )
    for command in (
        f"git show {BUG_SHA[:7]}",
        f"git show {HEAD_SHA[:7]}",
        f"git show {BUG_SHA[:7]} --stat",
        "git log --oneline --all",
        "git log -p",
    ):
        result = ground(command)
        assert result.exact_output is None, command
        for leak in ("Bug Patch", "Remove F2P Tests", "return a + b", "test_add"):
            assert leak not in (result.context or ""), (command, leak)
    assert not any(path.startswith("/repos/swesmith/") and "/commits" in path for path in calls)
    assert calls.count("/repos/BurntSushi/ripgrep/commits/3b7fd442") == 1


def _smith_service(tmp_path, monkeypatch, mirror: str):
    service = make_service(tmp_path)
    snapshot = make_snapshot(service, {"a.py": "def add(a, b):\n    return a - b\n"})
    monkeypatch.setattr(service, "_resolve_sha", lambda ref: ("swesmith", mirror, HEAD_SHA))
    monkeypatch.setattr(service, "_ensure_snapshot", lambda owner, repo, sha: snapshot)
    return service


def test_smith_mirror_shows_the_upstream_commit_it_was_built_from(tmp_path, monkeypatch):
    service = _smith_service(tmp_path, monkeypatch, "BurntSushi__ripgrep.3b7fd442")
    upstream_sha = "3b7fd442" + "9" * 32
    lookup = "/repos/BurntSushi/ripgrep/commits/3b7fd442"
    calls = _fake_github(
        monkeypatch,
        service,
        [(HEAD_SHA, "Remove F2P Tests"), (BUG_SHA, "Bug Patch"), (INITIAL_SHA, "Initial commit")],
        {BUG_SHA: _format_patch(BUG_SHA, "Bug Patch", _BUG_DIFF)},
        upstream={lookup: {"sha": upstream_sha, "commit": {"message": "Release 14.1.0\n\nnotes"}}},
    )

    def ground(command):
        return service.repo_context_for_instance("mini-coder", SMITH_ID, f"```bash\n{command}\n```")

    assert ground("git log --oneline -5").exact_output == f"{upstream_sha[:7]} Release 14.1.0"
    assert ground("git log --oneline -- a.py").exact_output == f"{upstream_sha[:7]} Release 14.1.0"
    assert ground("git rev-parse HEAD").exact_output == upstream_sha
    for command in (f"git show {upstream_sha[:7]}", f"git show {BUG_SHA[:7]}", "git show HEAD"):
        result = ground(command)
        assert result.exact_output is None, command
        for leak in ("Bug Patch", "Remove F2P Tests", "return a + b", "swesmith"):
            assert leak not in (result.context or ""), (command, leak)
    assert calls.count(lookup) == 1


@pytest.mark.parametrize(("age_days", "looked_up"), [(2, False), (6.9, False), (7.1, True)])
def test_a_missing_upstream_commit_is_remembered_for_seven_days(
    tmp_path, monkeypatch, age_days, looked_up
):
    service = _smith_service(tmp_path, monkeypatch, "BurntSushi__ripgrep.3b7fd442")
    marker = service.cache_dir / "upstream" / "BurntSushi__ripgrep__3b7fd442.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    failed_at = time.time() - age_days * 24 * 3600
    marker.write_text(json.dumps({"sha": None, "failed_at": failed_at, "kind": "absent"}))
    calls = _fake_github(monkeypatch, service, [], {})
    service.repo_context_for_instance("mini-coder", SMITH_ID, "```bash\ngit log --oneline\n```")
    assert ("/repos/BurntSushi/ripgrep/commits/3b7fd442" in calls) is looked_up


@pytest.mark.parametrize(
    ("message", "shows_upstream"),
    [("Merge pull request #329 from x/fix", False), ("Merge pull request #3290 from x/y", True)],
)
def test_smith_pr_mirror_never_shows_an_upstream_commit_naming_its_pr(
    tmp_path, monkeypatch, message, shows_upstream
):
    service = _smith_service(tmp_path, monkeypatch, "jawah__charset_normalizer.1fdd6463")
    upstream_sha = "1fdd6463" + "8" * 32
    lookup = "/repos/jawah/charset_normalizer/commits/1fdd6463"
    _fake_github(
        monkeypatch,
        service,
        [],
        {},
        upstream={lookup: {"sha": upstream_sha, "commit": {"message": message}}},
    )
    result = service.repo_context_for_instance(
        "mini-coder", "jawah__charset_normalizer.1fdd6463.pr_329", "```bash\ngit log --oneline\n```"
    )
    squashed = f"{HEAD_SHA[:7]} Initial commit"
    assert result.exact_output == (f"{upstream_sha[:7]} {message}" if shows_upstream else squashed)


def test_git_show_renders_only_commits_from_the_served_history(tmp_path, monkeypatch):
    service = make_service(tmp_path)
    make_snapshot(service, {"a.py": "def add(a, b):\n    return a - b\n"})
    monkeypatch.setattr(service, "_resolve_sha", lambda ref: ("o", "r", FULL_SHA))
    older, fix = "6" * 40, "7" * 40
    calls = _fake_github(
        monkeypatch,
        service,
        [(FULL_SHA, "Base"), (older, "Older change")],
        {
            older: _format_patch(older, "Older change", _BUG_DIFF),
            fix: _format_patch(fix, "Fix the reported bug", _BUG_DIFF),
        },
    )

    def ground(command):
        return service.repo_context_for_instance("swe-zero", "o__r-1", f"```bash\n{command}\n```")

    served = ground(f"git show {older[:7]}")
    assert served.exact_output is not None
    assert "Older change" in served.exact_output

    refused = ground(f"git show {fix[:7]}")
    assert refused.exact_output is None
    assert "Fix the reported bug" not in (refused.context or "")
    assert not any(path.endswith(f"/commits/{fix[:7]}") for path in calls)
