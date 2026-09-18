# Any Anthropic-compatible client

Anything that honours `ANTHROPIC_BASE_URL` talks to the king over the Anthropic Messages API: the Anthropic SDKs,
Claude Code (own guides [here](../claude-code/cli.md)), and tools built on them.

## Requirements

- A client or SDK that lets you set the Anthropic base URL.
- Your key in the `sk-ant-api03-…` spelling (see [the main page](../README.md#create-an-account-and-get-a-key)).

## Values to paste

| Setting | Value |
|---|---|
| `ANTHROPIC_BASE_URL` | `https://api.albedo.tech` (no `/v1`; the SDK adds it) |
| Key | `sk-ant-api03-…` — as `ANTHROPIC_API_KEY` for the SDKs, as `ANTHROPIC_AUTH_TOKEN` for Claude Code |
| Model | `albedo-king` for chat; `albedo-king-agent` when the request carries a `Bash`-style shell tool |

Served routes: `POST /v1/messages` (streaming and non-streaming) and `POST /v1/messages/count_tokens`. The
`thinking` parameter is accepted and ignored; the king answers without extended thinking. `max_tokens` above
8,192 is lowered to 8,192.

## Example

```python
import anthropic

client = anthropic.Anthropic(base_url="https://api.albedo.tech", api_key="sk-ant-api03-…")
msg = client.messages.create(
    model="albedo-king",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Which king are you?"}],
)
print(msg.content[0].text)
```

With a shell tool the king drives a loop the way Claude Code does: declare one tool named `Bash` with a
`command` string, send the request with `model: albedo-king-agent`, run the returned `tool_use` command
yourself and send the output back as a `tool_result`. Other tools are ignored.

## Verify

```bash
curl -s https://api.albedo.tech/v1/messages \
  -H "x-api-key: sk-ant-api03-…" -H "anthropic-version: 2023-06-01" -H "content-type: application/json" \
  -d '{"model":"albedo-king","max_tokens":200,"messages":[{"role":"user","content":"Which king are you?"}]}'
```

returns a `message` whose text names the king.

## Known limits

- No images, documents or PDFs in `content`; text blocks only.
- Batches, Files and the Admin API do not exist here.
- Server-side tools (web search, code execution) are not available; only your own shell tool is honoured.

## Problems

Go back to [Troubleshooting on the main page](../README.md#troubleshooting). Anthropic-specific: `401` with a
key in `ANTHROPIC_API_KEY` from Claude Code means it wanted `ANTHROPIC_AUTH_TOKEN`; a `404` means the base URL
already ends in `/v1`.
