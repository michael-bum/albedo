import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bl = _load_script("benchmark_leaks")

_ISSUE = (
    "qml.simplify does not handle products with global phases correctly. Expected behavior: "
    "qml.simplify returns outputs equivalent to its input. Actual behavior: the global phase is "
    "dropped when the product contains two equal Pauli words on the same wire, so the simplified "
    "operator differs from the original by a factor of minus one."
)


def _benchmarks() -> dict:
    task = bl._task
    return {
        "swe-rebench-leaderboard": {
            "source": "t",
            "tasks": {"pennylaneai__pennylane-7671": task("PennyLaneAI/pennylane", 7671, _ISSUE)},
        },
        "aacr-bench": {
            "source": "t",
            "tasks": {"freecad-freecad#19411": task("FreeCAD/FreeCAD", 19411)},
        },
        "terminal-bench-2.1": {
            "source": "t",
            "tasks": {
                "fix-git": task(
                    text="I just made some changes to my personal site and checked "
                    "out master, but now I can't find those changes. Please help "
                    "me find them and merge them into master."
                )
            },
        },
    }


def test_parse_id_formats():
    assert bl.parse_id("ethereum__web3.py-3690") == ("ethereum/web3.py", 3690)
    assert bl.parse_id("modin-project__modin.8c7799fd.pr_7434") == ("modin-project/modin", 7434)
    assert bl.parse_id("pandas-dev__pandas.95280573.func_pm_op_swap__x1") is None
    assert bl.parse_id("marimo-team__marimo-8387_interface") == ("marimo-team/marimo", 8387)
    assert bl.parse_id("dotnet-aspnetcore-pr-62936-2f372e8ff137") == ("dotnet-aspnetcore", 62936)
    assert bl.parse_id("pandas-dev__pandas-dbf8aaf4a3f3b41e5c1a402473df5da43813948f") is None


def test_pr_keys_cross_format():
    assert bl.pr_key("FreeCAD/FreeCAD", 19411) == bl.pr_key("freecad-freecad", "19411")
    assert bl.row_pr_keys("wireservice_csvkit_pr783", "wireservice/csvkit") == {
        "wireservice-csvkit#783"
    }
    assert bl.norm_id("PennyLaneAI__pennylane-7671_interface") == "pennylaneai__pennylane-7671"


def test_index_match_keys():
    index = bl.Index(_benchmarks())
    keys = {
        (h["benchmark"], h["key"])
        for h in index.match(
            "PennyLaneAI__pennylane-7671",
            "PennyLaneAI/pennylane",
            f"<pr_description>\n{_ISSUE}\n</pr_description>\nfix it please",
        )
    }
    assert keys == {("swe-rebench-leaderboard", k) for k in ("id", "repo_pr", "text")}
    # Same repo, different PR, unrelated text: not a leak.
    assert index.match("pennylaneai__pennylane-7000", "PennyLaneAI/pennylane", "other bug") == []
    # AACR is matched on repo + PR from a SWE-style id.
    assert [h["key"] for h in index.match("FreeCAD__FreeCAD-19411", None, "")] == ["repo_pr"]
    text_hit = index.match(
        "x",
        None,
        "Please solve this issue: I just made some changes to my "
        "personal site and checked out master, but now I can't find those "
        "changes. Please help me find them and merge them into master.",
    )
    assert [(h["benchmark"], h["key"]) for h in text_hit] == [("terminal-bench-2.1", "text")]
    assert index.match("exercism__python-1", "exercism/python", "")[0]["key"] == "exercism_repo"


def test_text_boilerplate_shared_by_many_tasks_is_ignored():
    shared = "the quick brown fox jumps over the lazy dog again and again every day " * 3
    tasks = {f"t{i}": bl._task(text=f"{shared} unique{i}") for i in range(4)}
    index = bl.Index({"terminal-bench-2.1": {"source": "t", "tasks": tasks}})
    assert index.match_text(shared) == []


def test_scan_reports_hits_per_instance(tmp_path):
    rows = {
        "instance_id": ["pennylaneai__pennylane-7671", "owner__repo-1"],
        "repo": ["PennyLaneAI/pennylane", "owner/repo"],
        "messages": [
            [{"role": "system", "content": "s"}, {"role": "user", "content": _ISSUE}],
            [{"role": "user", "content": "unrelated task"}],
        ],
    }
    pq.write_table(pa.table(rows), tmp_path / "train-0.parquet")
    result = bl.scan([str(tmp_path / "*.parquet")], bl.Index(_benchmarks()), workers=1)
    assert result["rows"] == 2
    assert list(result["hits"]) == ["pennylaneai__pennylane-7671"]
    json.dumps(result)  # report must be serializable
