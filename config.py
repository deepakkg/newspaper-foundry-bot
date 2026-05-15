from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv


DEFAULT_ENV_PATH = Path(__file__).resolve().parent / ".env"
PROJECT_ROOT = Path(__file__).resolve().parent
TIME_PATTERN = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")


@dataclass(frozen=True)
class AppConfig:
    ollama_host: str
    ollama_model: str
    ollama_api_key: str | None
    topics: list[str]
    tones: list[str]
    max_tweet_chars: int
    max_retries: int
    timeout_seconds: int
    post_to_x: bool
    x_api_key: str | None
    x_api_key_secret: str | None
    x_access_token: str | None
    x_access_token_secret: str | None
    x_username: str | None
    telegram_bot_token: str | None
    telegram_chat_id: str | None
    log_file_path: Path
    run_timezone: str
    enabled_run_slots: list[str]


def _parse_csv_list(value: str, name: str) -> list[str]:
    items = [item.strip() for item in value.split(",")]
    cleaned = [item for item in items if item]
    if not cleaned:
        raise ValueError(f"{name} must contain at least one non-empty item.")
    return cleaned


def _parse_positive_int(value: str, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc

    if parsed <= 0:
        raise ValueError(f"{name} must be greater than 0.")
    return parsed


def _parse_bool(value: str, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"{name} must be a boolean-like value.")


def _parse_time_24h(value: str, name: str) -> str:
    normalized = value.strip()
    if not TIME_PATTERN.fullmatch(normalized):
        raise ValueError(f"{name} must be in HH:MM 24-hour format.")
    return normalized


def _parse_time_list(value: str, name: str) -> list[str]:
    return [_parse_time_24h(item, name) for item in _parse_csv_list(value, name)]


def _parse_timezone(value: str, name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty.")

    try:
        ZoneInfo(normalized)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"{name} must be a valid IANA timezone name.") from exc
    return normalized


def _read_required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def load_config(env_path: Path | None = None) -> AppConfig:
    resolved_env_path = env_path or DEFAULT_ENV_PATH
    load_dotenv(resolved_env_path, override=True)

    ollama_host = _read_required_env("OLLAMA_HOST")
    default_model = "gemma4:31b-cloud" if ollama_host == "https://ollama.com" else "gemma3:1b"
    ollama_model = os.getenv("OLLAMA_MODEL", default_model).strip() or default_model
    ollama_api_key = os.getenv("OLLAMA_API_KEY", "").strip() or None
    topics = _parse_csv_list(_read_required_env("TOPICS"), "TOPICS")
    tones = _parse_csv_list(_read_required_env("TONES"), "TONES")
    max_tweet_chars = _parse_positive_int(
        os.getenv("MAX_TWEET_CHARS", "230"), "MAX_TWEET_CHARS"
    )
    max_retries = _parse_positive_int(os.getenv("MAX_RETRIES", "5"), "MAX_RETRIES")
    timeout_seconds = _parse_positive_int(
        os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"), "OLLAMA_TIMEOUT_SECONDS"
    )
    post_to_x = _parse_bool(os.getenv("POST_TO_X", "true"), "POST_TO_X")
    x_api_key = os.getenv("X_API_KEY", "").strip() or None
    x_api_key_secret = os.getenv("X_API_KEY_SECRET", "").strip() or None
    x_access_token = os.getenv("X_ACCESS_TOKEN", "").strip() or None
    x_access_token_secret = os.getenv("X_ACCESS_TOKEN_SECRET", "").strip() or None
    x_username = os.getenv("X_USERNAME", "").strip() or None
    telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip() or None
    telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip() or None
    log_file_raw = (
        os.getenv("LOG_FILE_PATH", "logs/tweet-history.md").strip()
        or "logs/tweet-history.md"
    )
    run_timezone = _parse_timezone(
        os.getenv("RUN_TIMEZONE", "Asia/Kolkata"), "RUN_TIMEZONE"
    )
    enabled_run_slots = _parse_time_list(
        os.getenv("ENABLED_RUN_SLOTS", "06:00,10:00,14:00,18:00"),
        "ENABLED_RUN_SLOTS",
    )
    log_file_path = Path(log_file_raw)
    if not log_file_path.is_absolute():
        log_file_path = PROJECT_ROOT / log_file_path

    if post_to_x:
        missing = [
            name
            for name, value in (
                ("X_API_KEY", x_api_key),
                ("X_API_KEY_SECRET", x_api_key_secret),
                ("X_ACCESS_TOKEN", x_access_token),
                ("X_ACCESS_TOKEN_SECRET", x_access_token_secret),
                ("X_USERNAME", x_username),
            )
            if not value
        ]
        if missing:
            missing_str = ", ".join(missing)
            raise ValueError(
                f"POST_TO_X is enabled but required X credentials are missing: {missing_str}"
            )

    if ollama_host.startswith("https://") and ollama_api_key is None:
        raise ValueError(
            "OLLAMA_API_KEY is required when using a hosted Ollama endpoint."
        )

    if bool(telegram_bot_token) != bool(telegram_chat_id):
        raise ValueError(
            "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must both be set to enable Telegram delivery."
        )

    return AppConfig(
        ollama_host=ollama_host,
        ollama_model=ollama_model,
        ollama_api_key=ollama_api_key,
        topics=topics,
        tones=tones,
        max_tweet_chars=max_tweet_chars,
        max_retries=max_retries,
        timeout_seconds=timeout_seconds,
        post_to_x=post_to_x,
        x_api_key=x_api_key,
        x_api_key_secret=x_api_key_secret,
        x_access_token=x_access_token,
        x_access_token_secret=x_access_token_secret,
        x_username=x_username,
        telegram_bot_token=telegram_bot_token,
        telegram_chat_id=telegram_chat_id,
        log_file_path=log_file_path,
        run_timezone=run_timezone,
        enabled_run_slots=enabled_run_slots,
    )
