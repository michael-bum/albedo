import { KEYS_API_BASE, AGENT_API_BASE } from "../config.js";
import { el, mount } from "../dom.js";
import { fmtRelative, fmtCount } from "../format.js";

const $ = id => document.getElementById(id);
const params = new URLSearchParams(location.search);
function allowedBase(value) {
  if (!value) return null;
  try {
    const u = new URL(value);
    return u.protocol === "https:" && (u.hostname === "albedo.tech" || u.hostname.endsWith(".albedo.tech")) ? value : null;
  } catch { return null; }
}
const stored = key => allowedBase(localStorage.getItem(key));
const keysBase = (allowedBase(params.get("api")) || stored("keysApiBase") || KEYS_API_BASE).replace(/\/$/, "");
if (allowedBase(params.get("api"))) localStorage.setItem("keysApiBase", params.get("api"));
const agentBase = (allowedBase(params.get("agent")) || stored("agentApiBase") || AGENT_API_BASE).replace(/\/$/, "");
if (allowedBase(params.get("agent"))) localStorage.setItem("agentApiBase", params.get("agent"));

const KEY_PLACEHOLDER = "<your key>";
const CHAT_MODEL = "albedo-king";
const AGENT_MODEL = "albedo-king-agent";

let me = null;            // { account, keys, limits } or null when signed out
let service = null;       // /status payload once loaded (king roman for the tier card)
let revealed = null;      // { secrets, key } only right after create / rotate
let client = localStorage.getItem("keysSetupClient") || "claude";

async function api(path, { method = "GET", body } = {}) {
  const res = await fetch(`${keysBase}${path}`, {
    method,
    credentials: "include",
    headers: body ? { "content-type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  if (res.status === 401) return { status: 401, data: null };
  let data = null;
  try { data = await res.json(); } catch { data = null; }
  return { status: res.status, data };
}

// in-page two-step button: first click shows the warning with "yes" / "cancel" in place, no browser popups
function confirmButton(label, warning, onConfirm, cls = "") {
  const wrap = el("span", { class: "keys-confirm" });
  const arm = () => mount(wrap, button(label, () => {
    mount(wrap,
      el("span", { class: "keys-confirm-text" }, warning),
      button(`yes, ${label}`, () => { onConfirm(); arm(); }, cls || "danger"),
      button("cancel", arm));
    setTimeout(() => { if (wrap.querySelector(".keys-confirm-text")) arm(); }, 8000);
  }, cls));
  arm();
  return wrap;
}

function iso(seconds) {
  return seconds ? new Date(seconds * 1000).toISOString() : null;
}

function fmtDate(seconds) {
  if (!seconds) return "—";
  return new Date(seconds * 1000).toISOString().slice(0, 10);
}

function button(label, onClick, cls = "") {
  return el("button", { type: "button", class: `keys-btn ${cls}`.trim(), onClick }, label);
}

function copyButton(text, label = "copy") {
  const btn = el("button", { type: "button", class: "keys-copy" }, label);
  btn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(typeof text === "function" ? text() : text);
      btn.textContent = "copied";
      btn.classList.add("copied");
      setTimeout(() => { btn.textContent = label; btn.classList.remove("copied"); }, 1400);
    } catch {
      btn.textContent = "select & copy";
    }
  });
  return btn;
}

function kv(k, v, cls = "") {
  return el("div", { class: "kv" }, el("span", { class: "k" }, k), el("span", { class: `v ${cls}`.trim() }, v));
}

// ---- account ---------------------------------------------------------------------------------

function renderAccount() {
  const meta = $("account-meta");
  if (!me) {
    meta.textContent = "signed out";
    const next = encodeURIComponent(location.href.split("#")[0]);
    mount($("account-wrap"),
      el("div", { class: "keys-card keys-signin" },
        el("div", { class: "keys-signin-text" },
          el("div", { class: "keys-card-title" }, "sign in with GitHub"),
          el("div", { class: "keys-card-sub" },
            "GitHub identifies you and stops key farming. We only read your public profile and store the id and login. Accounts younger than 30 days cannot create keys.")),
        el("a", { class: "keys-btn primary", href: `${keysBase}/login/github?next=${next}` },
          githubIcon(), "continue with GitHub")));
    return;
  }
  const a = me.account;
  const limits = me.limits || {};
  meta.textContent = `${a.tier} tier`;
  mount($("account-wrap"),
    el("div", { class: "keys-card keys-account" },
      el("div", { class: "keys-account-id" },
        githubIcon(),
        el("a", { href: `https://github.com/${a.login}`, target: "_blank", rel: "noopener", class: "mono" }, a.login),
        el("span", { class: "pill live" }, a.tier)),
      el("div", { class: "keys-account-meta mono" }, `member since ${fmtDate(a.created_at)}`),
      button("sign out", async () => {
        await api("/logout", { method: "POST" });
        me = null; revealed = null;
        renderAll();
      }),
      confirmButton("delete account", "revokes every key, forgets your GitHub id, 24 h before you can sign in again",
        async () => {
          const { status, data } = await api("/account/delete", { method: "POST" });
          if (status >= 400) { flash(data?.error?.message || `request failed (${status})`); return; }
          me = null; revealed = null;
          renderAll();
          notice("account deleted · keys revoked · the same GitHub account can sign in again after 24 h");
        }, "danger")),
    renderTier(a, limits));
}

function notice(message) {
  const node = el("div", { class: "keys-flash ok" }, message);
  $("account-wrap").prepend(node);
  setTimeout(() => node.remove(), 8000);
}

const TIER_NOTES = {
  standard: "default tier for every GitHub account · verified SN97 hotkeys get the miner tier",
  internal: "internal tier · no key expiry · limits set by the team",
  miner: "verified SN97 miner · raised limits · 10 active keys, still 5 creations a day",
};

function discordLink(limits, text = "Albedo channel on the Bittensor Discord") {
  return limits?.support_url
    ? el("a", { href: limits.support_url, target: "_blank", rel: "noopener" }, text)
    : text;
}

function renderTier(account, limits) {
  const king = service?.king?.roman ? `ALBEDO-${service.king.roman}` : "—";
  return el("div", { class: "keys-card keys-tier" },
    el("div", { class: "keys-card-title" }, `account tier · ${account.tier}`),
    el("div", { class: "kv-grid" },
      kv("requests", `${limits.rpm ?? "—"} per minute · ${limits.parallel ?? "—"} parallel`),
      kv("completion tokens", limits.daily_completion_tokens ? `${fmtCount(limits.daily_completion_tokens)} per 24 h` : "—"),
      kv("prompt cap", limits.max_prompt_tokens ? `${fmtCount(limits.max_prompt_tokens)} tokens` : "no cap"),
      kv("reply cap", "8,192 tokens"),
      kv("keys", `${limits.max_keys ?? "—"} active · ${limits.creations_per_day ?? "—"} creations a day`),
      kv("key lifetime", limits.key_ttl_days ? `${limits.key_ttl_days} days` : "no expiry"),
      kv("king", king, "gold")),
    el("div", { class: "keys-card-sub" }, TIER_NOTES[account.tier] || "limits set by the team",
      " · need a higher limit? ask in the ", discordLink(limits), " and tell us how the king is working for you"));
}

function githubIcon() {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("width", "14"); svg.setAttribute("height", "14"); svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("fill", "currentColor"); svg.setAttribute("aria-hidden", "true");
  const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
  path.setAttribute("d", "M12 .5C5.65.5.5 5.65.5 12c0 5.08 3.29 9.4 7.86 10.93.58.1.79-.25.79-.55v-1.96c-3.2.7-3.88-1.54-3.88-1.54-.52-1.32-1.27-1.68-1.27-1.68-1.04-.71.08-.7.08-.7 1.15.08 1.76 1.18 1.76 1.18 1.03 1.76 2.7 1.25 3.36.96.1-.74.4-1.25.73-1.54-2.55-.29-5.24-1.27-5.24-5.66 0-1.25.45-2.28 1.18-3.08-.12-.29-.51-1.46.11-3.04 0 0 .97-.31 3.18 1.18.92-.26 1.91-.39 2.89-.4.98.01 1.97.14 2.89.4 2.2-1.49 3.17-1.18 3.17-1.18.63 1.58.24 2.75.12 3.04.74.8 1.18 1.83 1.18 3.08 0 4.4-2.7 5.36-5.27 5.65.41.36.78 1.06.78 2.13v3.16c0 .31.21.66.79.55C20.21 21.4 23.5 17.08 23.5 12 23.5 5.65 18.35.5 12 .5z");
  svg.append(path);
  return svg;
}

// ---- keys ------------------------------------------------------------------------------------

const STATUS_LABEL = { active: "active", expiring: "expiring", expired: "expired", revoked: "revoked" };

function keyStatus(key) {
  if (key.status === "active" && expiresSoon(key.expires_at)) return "expiring";
  return key.status;
}

function hintText(key, prefix = "ak-") {
  return `${prefix}${key.hint_head || "…"}…${key.hint}`;
}

function renderKeys() {
  const meta = $("key-meta");
  if (!me) {
    meta.textContent = "";
    mount($("key-wrap"), el("div", { class: "empty" }, "sign in to manage your keys"));
    return;
  }
  const keys = me.keys || [];
  const limits = me.limits || {};
  const active = keys.filter(k => k.status === "active").length;
  meta.textContent = `${active} active · ${limits.keys_left ?? "—"} more allowed · ${limits.creations_left ?? "—"} creations left today`;
  mount($("key-wrap"),
    revealed ? renderReveal() : null,
    renderCreate(limits),
    keys.length ? renderTable(keys) : el("div", { class: "empty" }, "no keys yet"),
    renderQuota(limits));
}

function renderCreate(limits) {
  const input = el("input", { type: "text", class: "keys-input mono", maxlength: "40", placeholder: "key name · e.g. laptop, desktop, ci" });
  const blocked = (limits.keys_left ?? 1) <= 0 ? `limit of ${limits.max_keys} active keys reached — revoke one first`
    : (limits.creations_left ?? 1) <= 0 ? "creation limit reached for today" : null;
  const submit = button("create key", () => {
    const name = input.value.trim();
    if (!name) { input.focus(); return; }
    mutateKey("/key", { name }).then(() => { input.value = ""; });
  }, "primary");
  if (blocked) submit.disabled = true;
  input.addEventListener("keydown", e => { if (e.key === "Enter") submit.click(); });
  return el("div", { class: "keys-card keys-create" },
    el("div", { class: "keys-card-title" }, "create a key"),
    el("div", { class: "keys-create-row" }, input, submit),
    el("div", { class: "keys-card-sub" }, blocked
      || `Your tier allows ${limits.max_keys ?? "—"} active keys and ${limits.creations_per_day ?? 5} creations a day. Names are unique among your active keys (case does not matter). The secret is shown once, right after creation.`));
}

function renderTable(keys) {
  const rows = keys.map(key => {
    const status = keyStatus(key);
    const live = key.status === "active";
    return el("tr", { class: live ? "" : "dim" },
      el("td", { class: "keys-name" }, key.name || "—"),
      el("td", { class: "mono keys-hint-cell" }, hintText(key)),
      el("td", { class: "mono" }, fmtDate(key.created_at)),
      el("td", { class: `mono ${status === "expiring" ? "bad" : ""}` }, key.expires_at ? fmtDate(key.expires_at) : "never"),
      el("td", { class: "mono" }, key.last_used_at ? fmtRelative(iso(key.last_used_at)) : "never"),
      el("td", { class: "mono" }, fmtCount(key.requests_7d || 0)),
      el("td", {}, el("span", { class: `pill ${live ? "live" : ""} ${status === "expiring" ? "warn" : ""}` }, STATUS_LABEL[status] || status)),
      el("td", { class: "keys-row-actions" }, live ? [
        confirmButton("rotate", "old secret stops now, new one shown once",
          () => mutateKey(`/key/${key.id}/rotate`), ""),
        confirmButton("revoke", "stops working immediately",
          () => mutateKey(`/key/${key.id}/revoke`, null, true), "danger"),
      ] : null));
  });
  return el("div", { class: "keys-table-wrap" },
    el("table", { class: "keys-table" },
      el("thead", {}, el("tr", {},
        el("th", {}, "name"), el("th", {}, "hint"), el("th", {}, "created"), el("th", {}, "expires"),
        el("th", {}, "last used"), el("th", {}, "requests · 7d"), el("th", {}, "status"), el("th", {}, ""))),
      el("tbody", {}, rows)),
    el("div", { class: "keys-table-foot mono" },
      `${keys.length} ${keys.length === 1 ? "key" : "keys"} · revoked keys stay listed for 24 h, rotated ones are replaced in place · the hint is the same in every spelling`));
}

function renderQuota(limits) {
  const dailyLimit = Number(limits.daily_completion_tokens) || 0;
  const today = Number(limits.tokens_24h) || 0;
  const ratio = dailyLimit ? Math.min(1, today / dailyLimit) : 0;
  return el("div", { class: "keys-card" },
    el("div", { class: "keys-quota" },
      el("div", { class: "keys-quota-head" },
        el("span", {}, "completion tokens · last 24 h · all keys"),
        el("span", { class: "mono" }, `${fmtCount(today)} / ${fmtCount(dailyLimit)}`)),
      el("div", { class: "keys-quota-bar" }, el("i", { style: `width:${(ratio * 100).toFixed(1)}%`, class: ratio > 0.9 ? "hot" : "" }))),
    ratio > 0.8 ? el("div", { class: "keys-card-sub" },
      "running low · the budget is a rolling 24 h window · need more? ask in the ", discordLink(limits), " and tell us how the king is working for you") : null);
}

function expiresSoon(seconds) {
  return seconds && seconds * 1000 - Date.now() < 14 * 86400 * 1000;
}

const SPELLINGS = [
  ["anthropic", "Claude Code, Anthropic SDKs"],
  ["openai", "Codex, Copilot, Cursor, OpenAI SDKs"],
  ["openai_project", "clients that insist on a project key"],
  ["albedo", "curl, aider, ACP and anything else"],
];

function renderReveal() {
  const { secrets, key } = revealed;
  return el("div", { class: "keys-card keys-reveal" },
    el("div", { class: "keys-card-title" }, `"${key.name}" · shown once`),
    el("div", { class: "keys-card-sub" }, "One secret, four spellings. Paste the one your client expects; they all unlock the same key. Store it now — once you leave this page only the hint stays visible. The setup snippets below already contain it."),
    el("div", { class: "keys-spellings" }, SPELLINGS.map(([fmt, who]) => secrets[fmt] ? el("div", { class: "keys-spelling" },
      el("div", { class: "keys-spelling-who" }, who),
      el("div", { class: "keys-secret" }, el("code", { class: "mono" }, secrets[fmt]), copyButton(secrets[fmt]))) : null)),
    el("div", { class: "keys-actions" }, button("done, hide it", () => { revealed = null; renderAll(); })));
}

async function mutateKey(path, body = null, revoke = false) {
  const { status, data } = await api(path, { method: "POST", body: body || undefined });
  if (status === 401) { me = null; renderAll(); return; }
  if (status >= 400) {
    flash(data?.error?.message || data?.error || `request failed (${status})`);
    return;
  }
  revealed = revoke ? null : (data?.secrets ? { secrets: data.secrets, key: data.key } : null);
  await loadMe();
  renderAll();
}

function flash(message) {
  const node = el("div", { class: "keys-flash" }, message);
  $("key-wrap").prepend(node);
  setTimeout(() => node.remove(), 5000);
}

// ---- setup -----------------------------------------------------------------------------------

// the context size a client is told must be the tier's prompt cap, not the engine's 262k:
// clients compact when they approach it, and the gateway refuses anything above it
const DEFAULT_PROMPT_CAP = 131072;
const contextTokens = () => Number(me?.limits?.max_prompt_tokens) || DEFAULT_PROMPT_CAP;

const CLIENTS = [
  { id: "claude", label: "Claude Code", spelling: "anthropic", snippet: key => ({ lang: "bash", text:
`export ANTHROPIC_BASE_URL=${agentBase}
export ANTHROPIC_AUTH_TOKEN=${key}
export ANTHROPIC_MODEL=${AGENT_MODEL}
export ANTHROPIC_DEFAULT_OPUS_MODEL=${AGENT_MODEL}
export ANTHROPIC_DEFAULT_SONNET_MODEL=${AGENT_MODEL}
export ANTHROPIC_DEFAULT_HAIKU_MODEL=${AGENT_MODEL}
export CLAUDE_CODE_MAX_CONTEXT_TOKENS=${contextTokens()}
claude --model ${AGENT_MODEL}`,
    note: `Agent mode: the king's shell command arrives as a Bash tool call, so Claude Code asks for permission as usual. Use ${CHAT_MODEL} for chat only. For one repo put the same variables in .claude/settings.local.json under "env". CLAUDE_CODE_MAX_CONTEXT_TOKENS is your tier's prompt cap (${fmtCount(contextTokens())}), so Claude Code compacts before a request is refused.` }) },
  { id: "codex", label: "Codex", spelling: "openai", snippet: key => ({ lang: "toml", file: "~/.codex/config.toml", text:
`model = "${AGENT_MODEL}"
model_provider = "albedo"
model_context_window = ${contextTokens()}
model_max_output_tokens = 8192

[model_providers.albedo]
name = "Albedo"
base_url = "${agentBase}/v1"
wire_api = "responses"
http_headers = { Authorization = "Bearer ${key}" }`,
    note: `Works for the Codex CLI and the Codex VS Code extension. model_context_window is your tier's prompt cap (${fmtCount(contextTokens())}), so Codex compacts in time. The /model picker only lists OpenAI's catalog; /status shows the real model.` }) },
  { id: "copilot", label: "Copilot · VS Code", spelling: "openai", snippet: key => ({ lang: "text", text:
`Chat view → model picker → Manage Language Models → add a custom endpoint
  URL:             ${agentBase}/v1
  models:          ${CHAT_MODEL}        (chat, tool calling off)
                   ${AGENT_MODEL}  (agent, tool calling on)
  maxInputTokens:  ${contextTokens()}   (your tier's prompt cap; in chatLanguageModels.json)
  maxOutputTokens: 8192
  API key:         ${key}
    enter it via "Update API Key" on the group — a key pasted into the JSON file is ignored`,
    note: "Agent mode runs the king's loop through Copilot's terminal tool; each step is one approved command." }) },
  { id: "cursor", label: "Cursor", spelling: "openai", snippet: key => ({ lang: "text", text:
`Settings → Models → OpenAI API Key
  Override OpenAI Base URL:  ${agentBase}/v1
  API key:                   ${key}
  custom model:              ${CHAT_MODEL}
  add ${CHAT_MODEL} as a custom model and pick it in the editor chat (Ctrl+L)`,
    note: "Cursor requires a Pro plan for custom models. Agent mode runs the king's commands through Cursor's Shell tool with the usual approval." }) },
  { id: "acp", label: "ACP · Zed & others", spelling: "albedo", snippet: key => ({ lang: "bash", text:
`export ALBEDO_API_KEY=${key}
export ALBEDO_BASE_URL=${agentBase}
uvx albedo-king-acp`,
    note: "Any Agent Client Protocol client can launch this command as an agent. Commands run as subprocesses with per-command approval." }) },
  { id: "curl", label: "curl", spelling: "albedo", snippet: key => ({ lang: "bash", text:
`curl ${agentBase}/v1/chat/completions \\
  -H "Authorization: Bearer ${key}" \\
  -H "Content-Type: application/json" \\
  -d '{"model":"${CHAT_MODEL}","messages":[{"role":"user","content":"What does a Bittensor validator do?"}]}'`,
    note: "Also served: /v1/messages (Anthropic), /v1/responses (OpenAI Responses), /v1/models, /status." }) },
];

function renderSetup() {
  const current = CLIENTS.find(c => c.id === client) || CLIENTS[0];
  const key = revealed?.secrets?.[current.spelling] || KEY_PLACEHOLDER;
  const snippet = current.snippet(key);
  mount($("setup-wrap"),
    el("div", { class: "keys-tabs" }, CLIENTS.map(c =>
      el("button", { type: "button", class: `keys-tab${c.id === current.id ? " active" : ""}`, onClick: () => {
        client = c.id; localStorage.setItem("keysSetupClient", client); renderSetup();
      } }, c.label))),
    el("div", { class: "keys-snippet" },
      el("div", { class: "keys-snippet-head" },
        el("span", { class: "mono" }, snippet.file || snippet.lang),
        copyButton(snippet.text)),
      el("pre", {}, el("code", {}, snippet.text)),
      el("div", { class: "keys-card-sub" }, snippet.note)),
    revealed ? null : el("div", { class: "keys-hint" }, `${KEY_PLACEHOLDER} is replaced by your key, in the spelling this client expects, right after you create or rotate it.`));
}

// ---- service ---------------------------------------------------------------------------------

async function renderService() {
  const host = agentBase.replace(/^https?:\/\//, "");
  $("api-host").textContent = host;
  let status = null;
  try {
    const res = await fetch(`${agentBase}/status`, { cache: "no-store" });
    if (res.ok) status = await res.json();
  } catch { status = null; }
  service = status;
  if (me) renderAccount();
  if (!status) {
    $("service-state").textContent = "";
    $("service-meta").textContent = host;
    mount($("service-wrap"), el("div", { class: "empty" }, "status unavailable"));
    return;
  }
  const serving = status.state === "serving";
  $("service-state").textContent = serving ? "serving" : status.state || "";
  $("service-meta").textContent = host;
  const king = status.king || {};
  mount($("service-wrap"),
    el("div", { class: "kv-grid" },
      kv("state", status.state || "—", serving ? "ok" : "bad"),
      kv("king", king.roman ? `ALBEDO-${king.roman}` : king.name || "—", "gold"),
      kv("chat model", status.model || CHAT_MODEL),
      kv("agent model", AGENT_MODEL),
      kv("context", status.max_model_len ? `${fmtCount(status.max_model_len)} tokens` : "—")),
    status.notice ? el("div", { class: "note", style: "margin-top:12px" }, status.notice) : null);
}

function renderRules() {
  mount($("rules-wrap"),
    el("div", {}, "5 active named keys on the standard tier, 10 for verified miners · 5 creations a day on every tier · keys expire after 90 days and are recreated from this page"),
    el("div", {}, "limits are counted on the account, not the key: standard tier 30 requests per minute · 2 parallel · 200k completion tokens per day · 128k prompt tokens"),
    el("div", {}, "need a higher token limit? ask in the Albedo channel on the Bittensor Discord — and share your feedback about the king there: what it solved, where it failed, which client you use"),
    el("div", {}, "verified SN97 hotkeys get the miner tier — same place"),
    el("div", {}, "we store your GitHub id and login, hashed keys and per-request counts with a daily-salted ip hash · no email, no GitHub token, no prompts"));
}

// ---- boot ------------------------------------------------------------------------------------

async function loadMe() {
  const { status, data } = await api("/me");
  me = status === 200 && data?.account ? data : null;
}

function renderAll() {
  renderAccount();
  renderKeys();
  renderSetup();
}

(async () => {
  renderRules();
  renderService();
  try {
    await loadMe();
  } catch {
    me = null;
  }
  renderAll();
  if (!me && params.get("signin") === "failed") flash(params.get("reason") || "sign-in failed");
})();
