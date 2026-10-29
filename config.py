"""Configuration loading for the Telegram <-> Ollama bot."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent


def find_pi() -> str:
    """The systemd unit has a minimal PATH, so look in the usual install spots."""
    for candidate in (
        shutil.which("pi"),
        str(Path.home() / ".npm-global/bin/pi"),
        "/usr/local/bin/pi",
        "/opt/homebrew/bin/pi",
    ):
        if candidate and Path(candidate).exists():
            return candidate
    return "pi"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


def env_str(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def env_int(name: str, default: int) -> int:
    try:
        return int(env_str(name, str(default)))
    except ValueError:
        return default


def env_bool(name: str, default: bool) -> bool:
    return env_str(name, "1" if default else "0").lower() in ("1", "true", "yes", "on")


class Config:
    def __init__(self) -> None:
        self.token = env_str("TELEGRAM_BOT_TOKEN")
        self.owner_chat_id = env_str("OWNER_CHAT_ID")
        self.ollama_url = env_str("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
        self.model = env_str("OLLAMA_MODEL", "qwen3:8b")
        self.num_ctx = env_int("OLLAMA_NUM_CTX", 8192)
        self.num_predict = env_int("OLLAMA_NUM_PREDICT", 900)
        self.temperature = env_str("OLLAMA_TEMPERATURE", "")
        self.keep_alive = env_str("OLLAMA_KEEP_ALIVE", "-1")
        self.timeout = env_int("OLLAMA_TIMEOUT", 600)
        self.system_prompt = env_str(
            "SYSTEM_PROMPT",
            "You are Minamo, a helpful assistant running locally on the user's machine "
            "and reachable through Telegram. Be direct and concise. Reply in plain text; "
            "avoid Markdown tables and long code fences unless the user asks for code. "
            "If you are not sure about something, say so instead of guessing.",
        )
        self.history_messages = env_int("HISTORY_MESSAGES", 16)
        self.history_chars = env_int("HISTORY_CHARS", 12000)
        self.max_input_chars = env_int("MAX_INPUT_CHARS", 8000)
        self.edit_interval = float(env_str("EDIT_INTERVAL", "1.3"))
        self.stats_footer = env_bool("STATS_FOOTER", True)
        self.data_dir = Path(env_str("DATA_DIR", str(APP_DIR / "data")))
        self.log_file = env_str("LOG_FILE", str(self.data_dir / "bot.log"))
        self.debug = env_bool("DEBUG", False)

        # remote-build settings
        self.workspace_root = Path(
            env_str("WORKSPACE_ROOT", str(Path.home()))
        ).expanduser().resolve()
        self.shell_timeout = env_int("SHELL_TIMEOUT", 60)
        self.agent_auto = env_bool("AGENT_AUTO_APPROVE", False)

        # pi coding agent
        self.pi_bin = env_str("PI_BIN", find_pi())
        self.pi_provider = env_str("PI_PROVIDER", "ollama")
        self.pi_model = env_str("PI_MODEL", "qwen3:8b")
        self.pi_thinking = env_str("PI_THINKING", "high")
        self.pi_tools = env_str("PI_TOOLS", "read,ls,find,grep,bash,edit,write")
        self.pi_timeout = env_int("PI_TIMEOUT", 900)
        self.pi_approval_timeout = env_int("PI_APPROVAL_TIMEOUT", 900)
        self.approval_port = env_int("APPROVAL_PORT", 8787)

    def validate(self) -> None:
        if not self.token or ":" not in self.token:
            raise SystemExit(
                "TELEGRAM_BOT_TOKEN is missing. Put it in .env next to bot.py "
                "(see .env.example)."
            )
