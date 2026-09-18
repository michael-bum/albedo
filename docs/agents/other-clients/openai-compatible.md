# Any OpenAI-compatible client

Anything that lets you set an OpenAI base URL and key talks to the king: aider, OpenCode, Cline, Continue,
LiteLLM, the OpenAI SDKs, plain `curl`.

## Requirements

- A client with a configurable **base URL** (some call it *API base*, *endpoint*, *custom provider*).
- Your key in the `sk-…` spelling — or `sk-proj-…` if the client checks for an OpenAI project key (see
  [the main page](../README.md#create-an-account-and-get-a-key)).

## Values to paste

| Setting | Value |
|---|---|
| Base URL | `https://api.albedo.tech/v1` |
| API key | `sk-…` |
| Model | `albedo-king` for chat; `albedo-king-agent` when the client exposes a terminal / shell tool |

The endpoint serves `/v1/chat/completions` (streaming and non-streaming), `/v1/responses` and `/v1/models`.
Native tool calls are produced only for a **shell** tool (`run_in_terminal`, `run_terminal_cmd`,
`execute_command`, `exec_command`, `shell_command`, `bash`, `Shell`, …); other tools in the request are ignored.
Set `max_tokens` to 8,192 or less. Thinking/reasoning parameters are ignored.

## Examples

Python SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="https://api.albedo.tech/v1", api_key="sk-…")
r = client.chat.completions.create(
    model="albedo-king",
    messages=[{"role": "user", "content": "Which king are you?"}],
)
print(r.choices[0].message.content)
```

aider:

```bash
export OPENAI_API_BASE=https://api.albedo.tech/v1
export OPENAI_API_KEY=sk-…
aider --model openai/albedo-king
```

Continue (`~/.continue/config.yaml`):

```yaml
models:
  - name: Albedo king
    provider: openai
    model: albedo-king
    apiBase: https://api.albedo.tech/v1
    apiKey: sk-…
    roles: [chat]
```

OpenCode (`opencode.json`):

```json
{
  "provider": {
    "albedo": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Albedo king",
      "options": { "baseURL": "https://api.albedo.tech/v1", "apiKey": "sk-…" },
      "models": { "albedo-king-agent": { "name": "albedo-king-agent" } }
    }
  }
}
```

Cline / Roo Code: provider **OpenAI Compatible**, Base URL `https://api.albedo.tech/v1`, API key `sk-…`,
Model ID `albedo-king-agent`.

## Verify

```bash
curl -s https://api.albedo.tech/v1/models -H "Authorization: Bearer sk-…"
```

lists `albedo-king`, `albedo-king-<roman>` and `albedo-king-agent`. In the client, `Which king are you?`
returns a streamed answer naming the king.

## Known limits

- Clients whose agent mode relies on **file-edit or search tools** (not a shell) get text answers from the king
  but no edits. Clients with a shell tool (Cline's `execute_command`, OpenCode's `bash`) run the king's loop.
- `/v1/embeddings`, `/v1/images`, audio and fine-tuning routes do not exist. Point only the chat model at the
  king; keep embeddings on another provider.
- Model names other than the three above are accepted and served by the king; the reply reports the name you
  sent.

## Problems

Go back to [Troubleshooting on the main page](../README.md#troubleshooting). Most common here: a missing
`/v1` in the base URL (404) or `Authorization` sent without the `Bearer` prefix (401).
