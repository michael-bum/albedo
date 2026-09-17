const fs = require("fs");
const path = require("path");

function loadEnv() {
  const envPath = path.resolve(__dirname, "..", ".env");
  const env = { ...process.env };
  if (!fs.existsSync(envPath)) return env;
  for (const line of fs.readFileSync(envPath, "utf8").split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const index = trimmed.indexOf("=");
    if (index === -1) continue;
    const key = trimmed.slice(0, index);
    const value = trimmed.slice(index + 1);
    if (value !== "") env[key] = value;
  }
  return env;
}

const env = loadEnv();
delete env.KING_ENGINE_MODEL_PATH;
const root = path.resolve(__dirname, "..");

module.exports = {
  apps: [
    {
      name: "albedo-king-engine",
      cwd: root,
      script: ".venv/bin/python",
      args: "chat_to_king/engine/supervisor.py",
      autorestart: true,
      restart_delay: 30000,
      kill_timeout: 20000,
      env,
    },
    {
      name: "albedo-king-chat",
      cwd: root,
      script: ".venv/bin/python",
      args: "chat_to_king/web/main.py",
      autorestart: true,
      env,
    },
    {
      name: "albedo-king-agent",
      cwd: root,
      script: ".venv/bin/python",
      args: "chat_to_king/agent/main.py",
      autorestart: true,
      env,
    },
    {
      name: "albedo-king-token-rollup",
      cwd: root,
      script: ".venv/bin/python",
      args: "chat_to_king/token_rollup.py",
      autorestart: false,
      cron_restart: "5 */4 * * *",
      env,
    },
  ],
};
