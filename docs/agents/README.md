# The Albedo king in your coding tools

Use the current SN97 king — the model that is winning the Albedo subnet right now — as the model behind
Claude Code, Codex, GitHub Copilot, Cursor or any OpenAI- or Anthropic-compatible client. You keep your
editor, your terminal and your approval prompts; only the model changes.

Your client sends the request to `https://api.albedo.tech`. The gateway checks your key, translates the
client's tool protocol into the king's way of working (one shell command per turn) and forwards the request to
the vLLM engine that serves the current king. When a new king is crowned the gateway keeps answering while the
engine reloads.

## What we support

| Agent | Surface | Guide |
|---|---|---|
| Claude Code | terminal `claude` | [claude-code/cli.md](claude-code/cli.md) |
| Claude Code | VS Code extension | [claude-code/vscode-extension.md](claude-code/vscode-extension.md) |
| Codex | terminal `codex` | [codex/cli.md](codex/cli.md) |
| Codex | VS Code extension | [codex/vscode-extension.md](codex/vscode-extension.md) |
| GitHub Copilot Chat | VS Code (bring your own key) | [copilot/vscode.md](copilot/vscode.md) |
| Cursor | Cursor IDE (Pro plan) | [cursor/ide.md](cursor/ide.md) |
| Anything with an OpenAI-style base URL | aider, OpenCode, Cline, Continue, SDKs… | [other-clients/openai-compatible.md](other-clients/openai-compatible.md) |
| Anything that takes `ANTHROPIC_BASE_URL` | Anthropic SDKs and wrappers | [other-clients/anthropic-compatible.md](other-clients/anthropic-compatible.md) |

Every guide assumes you have read the three sections below: they hold the key, the connection values and the
limits, which are the same for all clients.

## Create an account and get a key

Keys are self-serve at **https://albedo.tech/keys.html**. Sign in with GitHub (we read only your public profile
and store the numeric id and login; accounts younger than 30 days cannot create keys), give the key a name such as
`laptop` or `ci`, and copy the secret. It is shown **once**. Afterwards the page shows only a hint
(`ak-071…0234`), the expiry and the usage.

You get **one** secret in four spellings. They are the same secret with a different prefix, so a client that
validates the shape of the key accepts it:

| Spelling | Paste it into |
|---|---|
| `ak-…` | curl, aider, OpenCode and any client that accepts an arbitrary key |
| `sk-ant-api03-…` | Claude Code, Anthropic SDKs |
| `sk-…` | Codex, GitHub Copilot, Cursor, OpenAI SDKs |
| `sk-proj-…` | clients that insist on an OpenAI *project* key |

Whichever spelling you paste, usage and limits are counted on your account, across all your keys.

- **Several keys:** 5 active named keys on the standard tier (10 for verified miners), so you can revoke the laptop
  key without touching the desktop one. Five creations a day on every tier.
- **Expiry:** standard-tier keys are valid for 90 days. After that every request returns `401`; create a new one on the page.
- **Rotate or revoke:** on the page, per key. Rotating keeps the name, revokes the old secret at once and shows the
  new one once. Revoking one spelling revokes all four.
- **Keep it private.** Anyone holding the key spends your daily quota.
- **Delete account:** on the page. Every key is revoked at once and your GitHub login is removed; only anonymous
  request counts and a keyed hash of your GitHub id remain (it enforces the wait below). The same GitHub account can sign in again after 24 hours and starts fresh;
  the wait exists so that deleting an account never resets a daily limit.

## Connection values

| Value | |
|---|---|
| Base URL | `https://api.albedo.tech` (OpenAI-style clients usually want `https://api.albedo.tech/v1`) |
| Auth header | `Authorization: Bearer <key>` or `x-api-key: <key>` |
| Endpoints | `POST /v1/chat/completions`, `POST /v1/completions`, `POST /v1/messages` (+ `/v1/messages/count_tokens`), `POST /v1/responses`, `GET /v1/models` |
| Public, no key | `GET /status`, `GET /health` |

Model names:

| Model | Use it for |
|---|---|
| `albedo-king` | Chat. Ask questions, paste code, get answers. The king does not run anything. |
| `albedo-king-agent` | Agent loops. The king answers with one shell command per turn; the gateway hands it to your client's terminal tool (`Bash`, `Shell`, `run_in_terminal`, `exec_command`, …), your client asks you for approval, runs it and sends the output back. |
| `albedo-king-<roman>` | The same as `albedo-king`, named after the king currently served (e.g. `albedo-king-cxxv`). Handy when you want to see which king answered. |

Any request that carries a recognised shell tool goes through the agent path whatever model name it uses, so
Cursor's built-in Agent mode works with plain `albedo-king`. Unknown model names are served by the king too.

## Limits

| Tier | Requests / min | Parallel requests | Completion tokens / day | Prompt cap | Reply cap | Key lifetime |
|---|---|---|---|---|---|---|
| standard | 30 | 2 | 200,000 | 131,072 tokens | 8,192 tokens | 90 days |
| miner (verified SN97 hotkey) | 120 | 4 | 1,000,000 | 262,144 tokens | 8,192 tokens | 180 days |

- Only **completion** tokens count against the daily quota. Agent clients replay the whole conversation on every
  step; those prompt tokens are free for you.
- The reply cap is applied to every request: a larger `max_tokens` is lowered to 8,192.
- The prompt cap is the per-request budget of your tier and the gateway refuses anything above it. **Tell your
  client that number as its context size** (`CLAUDE_CODE_MAX_CONTEXT_TOKENS`, Codex `model_context_window`,
  Copilot `maxInputTokens`; the keys page fills it in for your tier), not the engine's 262,144: Claude Code, Codex
  and Copilot compact or trim when they near the number they were told, so a larger value ends the session with a
  `400` instead of a compaction. Cursor manages context itself; when it reports the cap, start a new chat.
- **Need more?** Ask for a higher limit in the Albedo channel on the Bittensor Discord, and tell us there how the king is working for you:
  what it solved, where it failed, which client you use. That feedback shapes the next king. SN97 miners get the miner tier by
  posting their hotkey there; other tiers exist for validators and partners, ask in the same place.

A rejected request is an HTTP `429` with a `retry-after` header (seconds) and a JSON body:

```json
{"error": {"message": "Rate limit (rpm). Retry after 12s.", "type": "rate_limit_exceeded", "code": "rate_limit_exceeded"}}
```

`Rate limit (parallel)` means two of your requests are already running; `Daily token quota exhausted. Need a
higher limit? Ask in the Albedo channel on the Bittensor Discord and tell us how the king is working for you.`
(code `quota_exceeded`, `retry-after: 3600`) means the 24-hour completion budget is used up. IDE clients show the
message text in the chat.

## What to expect

- **The king works through your shell tool only.** It reads files with `cat`, `grep`, `sed`, `git`, and writes
  them with heredocs (or `Set-Content` when it detects PowerShell). It does not use your client's file-edit,
  search or browser tools, has no MCP servers and no web access of its own; a request to "look something up
  online" is answered with what your shell can reach (`curl`), or declined.
- **One command per turn.** Each command shows up in your client's normal approval prompt. Your client's own
  system prompt, instructions files (`CLAUDE.md`, `AGENTS.md`, `.cursorrules`) and attached files are passed to
  the king untouched.
- **Loop guard.** If the king repeats the same command with the same result the gateway tells it so, and after
  four repeats it ends the turn with a visible ⚠️ message. Rephrase the task and continue.
- **New king loading.** When a new king is crowned the engine restarts (a few minutes). Meanwhile every request
  is answered with a normal completion whose text is `👑 King <roman> is loading on the GPUs (~1–2 min). Please
  resend your message shortly.` plus the headers `x-albedo-status: loading` and `retry-after: 60`. Resend later.
- **Thinking is off.** The king answers directly; reasoning tokens are not generated or billed.
- **Sampling** follows the king's own generation settings (temperature 1.0) unless your client sets its own.

## Quick check

List the models (any of the four key spellings works):

```bash
curl -s https://api.albedo.tech/v1/models -H "Authorization: Bearer $ALBEDO_API_KEY"
```

You should see `albedo-king`, `albedo-king-<roman>` and `albedo-king-agent`, each with a `king` object naming
the served king and its HuggingFace repository. Then one prompt:

```bash
curl -s https://api.albedo.tech/v1/chat/completions \
  -H "Authorization: Bearer $ALBEDO_API_KEY" -H "content-type: application/json" \
  -d '{"model":"albedo-king","messages":[{"role":"user","content":"Which king are you, and what are you good at?"}]}'
```

In any configured client, type `Which king are you?` — a streamed answer within a couple of seconds means
everything is wired up (Codex is the exception: its instructions make the king answer as Codex, so use the task
below there). To try the agent path give it a task in a scratch folder: `Create hello.txt containing
hi, then show it.` You should be asked to approve one shell command, then see the file content.

## Troubleshooting

| Symptom | Cause → fix |
|---|---|
| `401` `Missing API key.` | The client did not send the key. Check the variable / field name the client page names (`ANTHROPIC_AUTH_TOKEN` vs `ANTHROPIC_API_KEY`, `env_key` vs `http_headers`, Copilot secret not re-entered). |
| `401` `Invalid, expired or revoked API key.` | Typo, a revoked key, or the 90 days are over. Paste the key again or ask for a new one. |
| `429` `Rate limit (rpm)` / `(parallel)` | You are above 30 requests per minute or already have 2 running. Wait for `retry-after`. Close a second agent session. |
| `429` `Daily token quota exhausted.` | 200,000 completion tokens used in the last 24 h. Comes back by itself. |
| Reply says `👑 King … is loading` | A new king is being loaded. Resend in one or two minutes; `GET /status` shows `state`. |
| "I don't have internet access" / king refuses to browse | Expected: no browser, no MCP. Ask it to use `curl` in your shell, or paste the content. |
| Client shows text like `I'll check the files…` but nothing runs | The king stopped after a preamble and the gateway's retry did not recover. Say `go ahead` or restate the task. |
| The king writes bash on Windows PowerShell | The gateway detects PowerShell from the tool description or Cursor's shell info. If your client hides the shell, tell the king in the first message: `You are in Windows PowerShell 5.` |
| PowerShell: `curl` errors, quoting breaks | `curl` is an alias of `Invoke-WebRequest` there. Ask for `Invoke-RestMethod -Uri …`. |
| Model picker in the client lists only vendor models | Normal for Codex and Claude Code: the picker shows the vendor catalogue, the request still goes to the king. `/status` (Codex) or the reply header shows the real model. |
| Model not offered at all | Cursor free plan (needs Pro), or the custom model was not added by name. See the client page. |
| `413 Request body too large.` | Bodies over 2 MB are refused. Detach large files. |

Per-client problems are on the client pages.

## Privacy

- We store **usage counts** per key: timestamp, endpoint, client family (from the user agent), HTTP status,
  prompt and completion token counts, latency, and a per-day salted hash of your IP for abuse detection.
- We do **not** store prompts, completions or file contents. Full request traces can be switched on temporarily
  for debugging on the serving host; they are deleted afterwards and are never shared.
- Requests travel through Cloudflare (TLS) to the serving host; the model runs on our GPUs, nothing is sent to a
  third-party model provider.
