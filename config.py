from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_ENV_PATH = Path(__file__).resolve().parent / ".env"
PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class AppConfig:
    llm_base_url: str
    llm_model: str
    llm_api_key: str | None
    topics: list[str]
    tones: list[str]
    max_tweet_chars: int
    max_retries: int
    timeout_seconds: int
    post_to_bluesky: bool
    bluesky_handle: str | None
    bluesky_app_password: str | None
    bluesky_service_url: str
    post_to_x: bool
    x_api_key: str | None
    x_api_key_secret: str | None
    x_access_token: str | None
    x_access_token_secret: str | None
    x_username: str | None
    telegram_notifications_enabled: bool
    telegram_bot_token: str | None
    telegram_chat_id: str | None
    discord_notifications_enabled: bool
    discord_webhook_url: str | None
    discord_bot_token: str | None
    discord_channel_id: str | None
    discord_approver_user_ids: list[str]
    approval_required: bool
    approval_timeout_minutes: int
    post_to_instagram: bool
    instagram_account_id: str | None
    instagram_access_token: str | None
    instagram_graph_base_url: str
    instagram_graph_api_version: str
    cloudinary_cloud_name: str | None
    cloudinary_api_key: str | None
    cloudinary_api_secret: str | None
    cloudinary_folder: str
    generated_image_dir: Path
    log_file_path: Path
    news_enabled: bool
    news_recency_hours: int
    news_region: str
    news_language: str


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


def _parse_non_empty_text(value: str, name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty.")
    return normalized


def _parse_optional_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _read_required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _normalize_llm_base_url(value: str) -> str:
    normalized = value.rstrip("/")
    if normalized == "https://ollama.com":
        raise ValueError(
            "LLM_BASE_URL is set to https://ollama.com, which is the Ollama "
            "website, not an OpenAI-compatible API endpoint. Use a provider "
            "base URL such as https://api.openai.com/v1 or local Ollama at "
            "http://localhost:11434/v1."
        )
    return normalized


def load_config(env_path: Path | None = None) -> AppConfig:
    resolved_env_path = env_path or DEFAULT_ENV_PATH
    load_dotenv(resolved_env_path, override=True)

    llm_base_url = _normalize_llm_base_url(_read_required_env("LLM_BASE_URL"))
    llm_model = _read_required_env("LLM_MODEL")
    llm_api_key = os.getenv("LLM_API_KEY", "").strip() or None
    topics = _parse_csv_list(_read_required_env("TOPICS"), "TOPICS")
    tones = _parse_csv_list(_read_required_env("TONES"), "TONES")
    max_tweet_chars = _parse_positive_int(
        os.getenv("MAX_TWEET_CHARS", "230"), "MAX_TWEET_CHARS"
    )
    max_retries = _parse_positive_int(os.getenv("MAX_RETRIES", "5"), "MAX_RETRIES")
    timeout_seconds = _parse_positive_int(
        os.getenv("LLM_TIMEOUT_SECONDS", "120"), "LLM_TIMEOUT_SECONDS"
    )
    post_to_bluesky = _parse_bool(os.getenv("POST_TO_BLUESKY", "false"), "POST_TO_BLUESKY")
    bluesky_handle = os.getenv("BLUESKY_HANDLE", "").strip() or None
    bluesky_app_password = os.getenv("BLUESKY_APP_PASSWORD", "").strip() or None
    bluesky_service_url = (
        os.getenv("BLUESKY_SERVICE_URL", "https://bsky.social").strip()
        or "https://bsky.social"
    ).rstrip("/")
    post_to_x = _parse_bool(os.getenv("POST_TO_X", "true"), "POST_TO_X")
    x_api_key = os.getenv("X_API_KEY", "").strip() or None
    x_api_key_secret = os.getenv("X_API_KEY_SECRET", "").strip() or None
    x_access_token = os.getenv("X_ACCESS_TOKEN", "").strip() or None
    x_access_token_secret = os.getenv("X_ACCESS_TOKEN_SECRET", "").strip() or None
    x_username = os.getenv("X_USERNAME", "").strip() or None
    telegram_notifications_enabled = _parse_bool(
        os.getenv("TELEGRAM_NOTIFICATIONS_ENABLED", "false"),
        "TELEGRAM_NOTIFICATIONS_ENABLED",
    )
    telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip() or None
    telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip() or None
    discord_notifications_enabled = _parse_bool(
        os.getenv("DISCORD_NOTIFICATIONS_ENABLED", "false"),
        "DISCORD_NOTIFICATIONS_ENABLED",
    )
    discord_webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "").strip() or None
    discord_bot_token = os.getenv("DISCORD_BOT_TOKEN", "").strip() or None
    discord_channel_id = os.getenv("DISCORD_CHANNEL_ID", "").strip() or None
    discord_approver_user_ids = _parse_optional_csv_list(
        os.getenv("DISCORD_APPROVER_USER_IDS", "")
    )
    approval_required = _parse_bool(
        os.getenv("APPROVAL_REQUIRED", "true"), "APPROVAL_REQUIRED"
    )
    approval_timeout_minutes = _parse_positive_int(
        os.getenv("APPROVAL_TIMEOUT_MINUTES", "90"), "APPROVAL_TIMEOUT_MINUTES"
    )
    post_to_instagram = _parse_bool(
        os.getenv("POST_TO_INSTAGRAM", "false"), "POST_TO_INSTAGRAM"
    )
    instagram_account_id = os.getenv("INSTAGRAM_ACCOUNT_ID", "").strip() or None
    instagram_access_token = os.getenv("INSTAGRAM_ACCESS_TOKEN", "").strip() or None
    instagram_graph_base_url = (
        os.getenv("INSTAGRAM_GRAPH_BASE_URL", "https://graph.instagram.com").strip()
        or "https://graph.instagram.com"
    ).rstrip("/")
    instagram_graph_api_version = (
        os.getenv("INSTAGRAM_GRAPH_API_VERSION", "v23.0").strip() or "v23.0"
    )
    cloudinary_cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME", "").strip() or None
    cloudinary_api_key = os.getenv("CLOUDINARY_API_KEY", "").strip() or None
    cloudinary_api_secret = os.getenv("CLOUDINARY_API_SECRET", "").strip() or None
    cloudinary_folder = (
        os.getenv("CLOUDINARY_FOLDER", "content-bot").strip() or "content-bot"
    )
    log_file_raw = (
        os.getenv("LOG_FILE_PATH", "logs/tweet-history.md").strip()
        or "logs/tweet-history.md"
    )
    generated_image_dir_raw = (
        os.getenv("GENERATED_IMAGE_DIR", "generated-posts").strip()
        or "generated-posts"
    )
    news_enabled = _parse_bool(os.getenv("NEWS_ENABLED", "true"), "NEWS_ENABLED")
    news_recency_hours = _parse_positive_int(
        os.getenv("NEWS_RECENCY_HOURS", "48"), "NEWS_RECENCY_HOURS"
    )
    news_region = _parse_non_empty_text(os.getenv("NEWS_REGION", "US"), "NEWS_REGION")
    news_language = _parse_non_empty_text(
        os.getenv("NEWS_LANGUAGE", "en"), "NEWS_LANGUAGE"
    )
    log_file_path = Path(log_file_raw)
    if not log_file_path.is_absolute():
        log_file_path = PROJECT_ROOT / log_file_path
    generated_image_dir = Path(generated_image_dir_raw)
    if not generated_image_dir.is_absolute():
        generated_image_dir = PROJECT_ROOT / generated_image_dir

    if post_to_bluesky:
        missing = [
            name
            for name, value in (
                ("BLUESKY_HANDLE", bluesky_handle),
                ("BLUESKY_APP_PASSWORD", bluesky_app_password),
            )
            if not value
        ]
        if missing:
            missing_str = ", ".join(missing)
            raise ValueError(
                "POST_TO_BLUESKY is enabled but required Bluesky credentials "
                f"are missing: {missing_str}"
            )

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

    if post_to_instagram:
        missing = [
            name
            for name, value in (
                ("INSTAGRAM_ACCOUNT_ID", instagram_account_id),
                ("INSTAGRAM_ACCESS_TOKEN", instagram_access_token),
                ("CLOUDINARY_CLOUD_NAME", cloudinary_cloud_name),
                ("CLOUDINARY_API_KEY", cloudinary_api_key),
                ("CLOUDINARY_API_SECRET", cloudinary_api_secret),
            )
            if not value
        ]
        if missing:
            missing_str = ", ".join(missing)
            raise ValueError(
                "POST_TO_INSTAGRAM is enabled but required Instagram/Cloudinary "
                f"credentials are missing: {missing_str}"
            )

    if approval_required and (post_to_bluesky or post_to_x or post_to_instagram):
        missing = [
            name
            for name, value in (
                ("DISCORD_BOT_TOKEN", discord_bot_token),
                ("DISCORD_CHANNEL_ID", discord_channel_id),
            )
            if not value
        ]
        if not discord_approver_user_ids:
            missing.append("DISCORD_APPROVER_USER_IDS")
        if missing:
            missing_str = ", ".join(missing)
            raise ValueError(
                "Publishing is enabled but required Discord approval settings "
                f"are missing: {missing_str}"
            )

    if llm_base_url.startswith("https://") and llm_api_key is None:
        raise ValueError("LLM_API_KEY is required when using a hosted LLM endpoint.")

    return AppConfig(
        llm_base_url=llm_base_url,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        topics=topics,
        tones=tones,
        max_tweet_chars=max_tweet_chars,
        max_retries=max_retries,
        timeout_seconds=timeout_seconds,
        post_to_bluesky=post_to_bluesky,
        bluesky_handle=bluesky_handle,
        bluesky_app_password=bluesky_app_password,
        bluesky_service_url=bluesky_service_url,
        post_to_x=post_to_x,
        x_api_key=x_api_key,
        x_api_key_secret=x_api_key_secret,
        x_access_token=x_access_token,
        x_access_token_secret=x_access_token_secret,
        x_username=x_username,
        telegram_notifications_enabled=telegram_notifications_enabled,
        telegram_bot_token=telegram_bot_token,
        telegram_chat_id=telegram_chat_id,
        discord_notifications_enabled=discord_notifications_enabled,
        discord_webhook_url=discord_webhook_url,
        discord_bot_token=discord_bot_token,
        discord_channel_id=discord_channel_id,
        discord_approver_user_ids=discord_approver_user_ids,
        approval_required=approval_required,
        approval_timeout_minutes=approval_timeout_minutes,
        post_to_instagram=post_to_instagram,
        instagram_account_id=instagram_account_id,
        instagram_access_token=instagram_access_token,
        instagram_graph_base_url=instagram_graph_base_url,
        instagram_graph_api_version=instagram_graph_api_version,
        cloudinary_cloud_name=cloudinary_cloud_name,
        cloudinary_api_key=cloudinary_api_key,
        cloudinary_api_secret=cloudinary_api_secret,
        cloudinary_folder=cloudinary_folder,
        generated_image_dir=generated_image_dir,
        log_file_path=log_file_path,
        news_enabled=news_enabled,
        news_recency_hours=news_recency_hours,
        news_region=news_region,
        news_language=news_language,
    )
