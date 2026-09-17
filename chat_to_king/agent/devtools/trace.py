#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
from pathlib import Path

from agent.protocols.parse import _responses_text, _text


def session_key(payload: dict, kind: str, client: str | None, account: str | None) -> str:
    ident = ""
    meta = payload.get("metadata")
    if isinstance(meta, dict) and isinstance(meta.get("user_id"), str):
        try:
            ident = str(json.loads(meta["user_id"]).get("session_id") or "")
        except (ValueError, AttributeError):
            ident = meta["user_id"]
    if not ident and isinstance(payload.get("prompt_cache_key"), str):
        ident = payload["prompt_cache_key"]
    if not ident and isinstance(payload.get("session_id"), str):
        ident = payload["session_id"]
    if not ident:
        ident = _first_user_text(payload, kind)
    raw = f"{client or ''}|{account or ''}|{ident}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _first_user_text(payload: dict, kind: str) -> str:
    items = payload.get("input") if kind == "responses" else payload.get("messages")
    if isinstance(items, str):
        return items
    for m in items or []:
        if isinstance(m, dict) and m.get("role") == "user":
            text = (
                _responses_text(m.get("content"))
                if kind == "responses"
                else _text(m.get("content"))
            )
            if text.strip():
                return text.strip()[:4000]
    return json.dumps(payload.get("prompt") or "")[:4000]


class Trace:
    def __init__(self, root: str) -> None:
        self.root = Path(root).expanduser() if root else None
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.root is not None

    @staticmethod
    def new(**fields) -> dict:
        return {"ts": time.time(), **fields, "king_requests": [], "king_responses": [], "notes": []}

    def write(self, turn: dict) -> None:
        if self.root is None:
            return
        turn["duration_ms"] = int((time.time() - turn.get("ts", time.time())) * 1000)
        session = turn.pop("session", None)
        client = turn.get("client") or "unknown"
        day = time.strftime("%Y-%m-%d", time.gmtime(turn.get("ts", time.time())))
        path = self.root / day / f"{client}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            if session is None:
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(turn, ensure_ascii=False, default=str) + "\n")
                return
            records = []
            if path.exists():
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        records.append(json.loads(line))
            for rec in records:
                if rec.get("session") == session:
                    rec["turns"].append(turn)
                    rec["last_ts"] = turn["ts"]
                    break
            else:
                records.append(
                    {
                        "session": session,
                        "client": client,
                        "account": turn.get("account"),
                        "user_agent": turn.get("user_agent"),
                        "started_ts": turn["ts"],
                        "last_ts": turn["ts"],
                        "turns": [turn],
                    }
                )
            tmp = path.with_suffix(".tmp")
            tmp.write_text(
                "".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in records),
                encoding="utf-8",
            )
            os.replace(tmp, path)


def _hms(ts: float) -> str:
    return time.strftime("%H:%M:%S", time.gmtime(ts))


def _show_turn(i: int, entry: dict, full: bool) -> None:
    print(
        f"--- turn {i} {_hms(entry['ts'])} {entry.get('path')} model={entry.get('model_requested')} "
        f"stream={entry.get('stream')} {entry.get('duration_ms')}ms notes={entry.get('notes')}"
    )
    payload = entry.get("client_payload") or {}
    msgs = payload.get("messages") or payload.get("input") or []
    tools = [
        t.get("name") or (t.get("function") or {}).get("name") for t in payload.get("tools") or []
    ]
    print(
        f"    client payload: {len(msgs)} messages/items, tools={tools[:12]}{'…' if len(tools) > 12 else ''}"
    )
    for req_i, req in enumerate(entry.get("king_requests") or []):
        kmsgs = req.get("messages") or []
        sampling = {
            k: v
            for k, v in req.items()
            if k in ("temperature", "top_p", "top_k", "max_tokens", "continue_final_message")
        }
        print(f"    king request {req_i}: {len(kmsgs)} messages, sampling={sampling}")
        for m in kmsgs if full else kmsgs[-2:]:
            content = (
                m.get("content")
                if isinstance(m.get("content"), str)
                else json.dumps(m.get("content"))
            )
            print(f"      [{m.get('role')}] {content if full else content[:400]}")
    for res_i, res in enumerate(entry.get("king_responses") or []):
        if isinstance(res, dict) and res.get("choices"):
            msg = res["choices"][0].get("message") or {}
            print(f"    king raw response {res_i} (usage={res.get('usage')}):")
            if msg.get("reasoning"):
                print(f"      <reasoning> {msg['reasoning']}")
            print(f"      {msg.get('content')}")
        else:
            print(f"    king raw response {res_i}: {json.dumps(res)[:2000]}")
    if entry.get("king_stream_raw"):
        raw = entry["king_stream_raw"]
        print(f"    king raw stream ({len(raw)} bytes): {raw if full else raw[:1500]}")
    if entry.get("reply") is not None:
        print(
            f"    reply to client: {json.dumps(entry['reply'], ensure_ascii=False)[: None if full else 2000]}"
        )


def _show(rec: dict, full: bool, last_turns: int) -> None:
    if "turns" not in rec:
        _show_turn(0, rec, full)
        return
    turns = rec["turns"]
    first = (turns[0].get("client_payload") or {}) if turns else {}
    kind = "responses" if "input" in first else "openai"
    query = _first_user_text(first, kind)[:200].replace("\n", " ")
    print(
        f"=== session {rec['session']} {rec.get('client')} account={rec.get('account')} "
        f"{_hms(rec['started_ts'])}–{_hms(rec['last_ts'])} turns={len(turns)}\n    first query: {query}"
    )
    shown = turns if (full or last_turns <= 0) else turns[-last_turns:]
    for i, t in enumerate(shown, start=len(turns) - len(shown) + 1):
        _show_turn(i, t, full)


def main() -> None:
    parser = argparse.ArgumentParser(prog="devtools.trace")
    parser.add_argument("file")
    parser.add_argument("-n", type=int, default=1, help="last N sessions")
    parser.add_argument(
        "--turns", type=int, default=0, help="only the last K turns of each session (0 = all)"
    )
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    lines = [ln for ln in Path(args.file).read_text(encoding="utf-8").splitlines() if ln.strip()]
    for line in lines[-args.n :]:
        _show(json.loads(line), args.full, args.turns)


if __name__ == "__main__":
    main()
