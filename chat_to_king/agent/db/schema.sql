CREATE TABLE IF NOT EXISTS tiers (
    name TEXT PRIMARY KEY,
    rpm INTEGER NOT NULL,
    parallel INTEGER NOT NULL,
    daily_completion_tokens BIGINT NOT NULL,
    max_prompt_tokens INTEGER NOT NULL,
    key_ttl_days INTEGER
);

CREATE TABLE IF NOT EXISTS accounts (
    id BIGSERIAL PRIMARY KEY,
    owner TEXT NOT NULL,
    contact TEXT,
    tier TEXT NOT NULL REFERENCES tiers(name),
    hotkey TEXT,
    created_at DOUBLE PRECISION NOT NULL,
    disabled_at DOUBLE PRECISION,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS key_formats (
    format TEXT PRIMARY KEY,
    prefix TEXT NOT NULL UNIQUE,
    enabled INTEGER NOT NULL DEFAULT 1,
    note TEXT
);

CREATE TABLE IF NOT EXISTS api_keys (
    id BIGSERIAL PRIMARY KEY,
    account_id BIGINT NOT NULL REFERENCES accounts(id),
    secret_hash TEXT NOT NULL UNIQUE,
    hint TEXT NOT NULL,
    label TEXT,
    created_at DOUBLE PRECISION NOT NULL,
    expires_at DOUBLE PRECISION,
    revoked_at DOUBLE PRECISION,
    rpm INTEGER,
    parallel INTEGER,
    daily_completion_tokens BIGINT,
    max_prompt_tokens INTEGER
);
CREATE INDEX IF NOT EXISTS api_keys_account ON api_keys (account_id);
CREATE INDEX IF NOT EXISTS api_keys_hint ON api_keys (hint);

CREATE TABLE IF NOT EXISTS usage (
    id BIGSERIAL PRIMARY KEY,
    ts DOUBLE PRECISION NOT NULL,
    key_id BIGINT NOT NULL REFERENCES api_keys(id),
    format TEXT,
    path TEXT NOT NULL,
    client TEXT,
    status INTEGER NOT NULL,
    stream INTEGER NOT NULL,
    prompt_tokens BIGINT NOT NULL,
    completion_tokens BIGINT NOT NULL,
    latency_ms INTEGER NOT NULL,
    ip_hash TEXT
);
CREATE INDEX IF NOT EXISTS usage_key_ts ON usage (key_id, ts);
CREATE INDEX IF NOT EXISTS usage_ts ON usage (ts);

CREATE TABLE IF NOT EXISTS events (
    id BIGSERIAL PRIMARY KEY,
    ts DOUBLE PRECISION NOT NULL,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    account_id BIGINT,
    key_id BIGINT,
    detail TEXT
);

INSERT INTO tiers (name, rpm, parallel, daily_completion_tokens, max_prompt_tokens, key_ttl_days) VALUES
    ('internal', 600, 8, 1000000000, 262144, NULL),
    ('beta', 30, 2, 200000, 131072, 90)
ON CONFLICT (name) DO NOTHING;

INSERT INTO key_formats (format, prefix, note) VALUES
    ('albedo', 'ak-', 'Albedo native'),
    ('anthropic', 'sk-ant-api03-', 'Claude Code, Anthropic SDKs'),
    ('openai', 'sk-', 'Codex, Copilot, Cursor, OpenAI SDKs'),
    ('openai_project', 'sk-proj-', 'OpenAI project-style keys')
ON CONFLICT (format) DO NOTHING;
