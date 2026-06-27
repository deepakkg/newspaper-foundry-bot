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
        "DISCORD_BOT_TOKEN": "test-discord-bot-token",
        "DISCORD_CHANNEL_ID": "1234567890",
        "DISCORD_APPROVER_USER_IDS": "111,222",
        "APPROVAL_REQUIRED": "true",
        "APPROVAL_TIMEOUT_MINUTES": "90",
        "POST_TO_INSTAGRAM": "false",
        "INSTAGRAM_ACCOUNT_ID": "",
        "INSTAGRAM_ACCESS_TOKEN": "",
        "INSTAGRAM_GRAPH_BASE_URL": "https://graph.instagram.com",
        "INSTAGRAM_GRAPH_API_VERSION": "v23.0",
        "CLOUDINARY_CLOUD_NAME": "",
        "CLOUDINARY_API_KEY": "",
        "CLOUDINARY_API_SECRET": "",
        "CLOUDINARY_FOLDER": "content-bot",
        "ARTICLE_LINKS_ENABLED": "false",
        "ARTICLE_LINKS_PAGE_URL": "",
        "ARTICLE_LINKS_MAX_ITEMS": "25",
        "GENERATED_IMAGE_DIR": str(Path(tempfile.gettempdir()) / "content-bot-test-images"),
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
