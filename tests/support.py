from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

from config import load_config


def write_env_file(path: Path, **overrides: str) -> None:
    values = {
        "LLM_BASE_URL": "http://localhost:11434/v1",
        "LLM_MODEL": "gemma3:1b",
        "TOPICS": "coffee,learning",
        "TONES": "witty,serious",
        "POST_TO_X": "false",
        "NEWS_ENABLED": "false",
        "NEWS_RECENCY_HOURS": "48",
        "NEWS_REGION": "US",
        "NEWS_LANGUAGE": "en",
        "POST_TO_BLUESKY": "false",
        "BLUESKY_HANDLE": "",
        "BLUESKY_APP_PASSWORD": "",
        "BLUESKY_SERVICE_URL": "https://bsky.social",
        "LLM_API_KEY": "",
        "X_API_KEY": "",
        "X_API_KEY_SECRET": "",
        "X_ACCESS_TOKEN": "",
        "X_ACCESS_TOKEN_SECRET": "",
        "X_USERNAME": "",
        "TELEGRAM_BOT_TOKEN": "",
        "TELEGRAM_CHAT_ID": "",
        "TELEGRAM_NOTIFICATIONS_ENABLED": "false",
        "DISCORD_NOTIFICATIONS_ENABLED": "false",
        "DISCORD_WEBHOOK_URL": "",
    }
    values.update(overrides)
    path.write_text(
        "\n".join(f"{key}={value}" for key, value in values.items()) + "\n",
        encoding="utf-8",
    )


def load_temp_config(**overrides: str):
    tmp_dir = tempfile.TemporaryDirectory()
    env_path = Path(tmp_dir.name) / ".env"
    overrides.setdefault("LOG_FILE_PATH", str(Path(tmp_dir.name) / "tweet-history.md"))
    write_env_file(env_path, **overrides)
    config = load_config(env_path)
    return tmp_dir, config


def chat_response(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text),
            )
        ]
    )
