from __future__ import annotations


def issue_key(store, owner: str, *, rpm: int, parallel: int, daily_tokens: int) -> str:
    account_id = store.add_account(owner, tier="standard")
    _key, shown = store.issue(
        account_id, rpm=rpm, parallel=parallel, daily_completion_tokens=daily_tokens
    )
    return shown["albedo"]


class FakeEngine:
    def __init__(self, serving: bool = True) -> None:
        self.serving = serving
        self.base_url = "http://engine"

    async def state(self):
        from common.engine_client import EngineState

        return EngineState.SERVING if self.serving else EngineState.LOADING
