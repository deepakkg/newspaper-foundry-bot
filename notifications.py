from __future__ import annotations

from datetime import timezone

from config import AppConfig
from discord_sender import send_discord_embed, send_discord_message
from logger import (
    PlatformLogResult,
    build_failure_telegram_summary,
    build_telegram_summary,
)
from news_fetcher import NewsItem
from telegram_sender import send_telegram_message


def format_news_published_at(news_item: NewsItem) -> str:
    return news_item.published_at.astimezone(timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )


def send_telegram_safely(
    config: AppConfig,
    message_text: str,
    *,
    warning_prefix: str,
) -> None:
    if not config.telegram_notifications_enabled:
        return
    if not config.telegram_bot_token or not config.telegram_chat_id:
        print(
            f"{warning_prefix}: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID "
            "must both be set."
        )
        return

    try:
        send_telegram_message(config, message_text)
    except Exception as exc:
        print(f"{warning_prefix}: {exc}")


def format_discord_field_value(value: str | None) -> str:
    return value if value else "Not available"


def format_platform_results(platform_results: list[PlatformLogResult]) -> str:
    lines: list[str] = []
    for result in platform_results:
        detail_parts = [result.status]
        if result.url:
            detail_parts.append(result.url)
        if result.identifier:
            detail_parts.append(result.identifier)
        if result.error:
            detail_parts.append(result.error)
        lines.append(f"{result.platform}: {' | '.join(detail_parts)}")
    return "\n".join(lines)


def build_discord_success_embed(
    *,
    topic: str,
    tone: str,
    tweet_text: str,
    time_taken_seconds: float,
    attempts: int,
    news_item: NewsItem | None,
    platform_results: list[PlatformLogResult] | None = None,
    partial: bool = False,
) -> dict[str, object]:
    fields: list[dict[str, object]] = [
        {"name": "Topic", "value": topic, "inline": True},
        {"name": "Tone", "value": tone, "inline": True},
        {"name": "Attempts", "value": str(attempts), "inline": True},
        {
            "name": "Time taken",
            "value": f"{time_taken_seconds:.2f} seconds",
            "inline": True,
        },
    ]
    if platform_results:
        fields.append(
            {
                "name": "Platform results",
                "value": format_platform_results(platform_results)[:1024],
                "inline": False,
            }
        )
    fields.append({"name": "Final post", "value": tweet_text, "inline": False})
    return {
        "title": "Post partially published" if partial else "Post published",
        "color": 0xF39C12 if partial else 0x2ECC71,
        "fields": fields,
    }


def build_discord_manual_embed(
    *,
    topic: str,
    tone: str,
    time_taken_seconds: float,
    attempts: int,
    news_item: NewsItem | None,
) -> dict[str, object]:
    fields: list[dict[str, object]] = [
        {"name": "Topic", "value": topic, "inline": True},
        {"name": "Tone", "value": tone, "inline": True},
        {"name": "Attempts", "value": str(attempts), "inline": True},
        {
            "name": "Time taken",
            "value": f"{time_taken_seconds:.2f} seconds",
            "inline": True,
        },
    ]
    if news_item:
        fields.extend(
            [
                {
                    "name": "News title",
                    "value": format_discord_field_value(news_item.title),
                    "inline": False,
                },
                {
                    "name": "News source",
                    "value": format_discord_field_value(news_item.source),
                    "inline": True,
                },
                {
                    "name": "News published",
                    "value": format_news_published_at(news_item),
                    "inline": True,
                },
            ]
        )
    return {"title": "Post ready", "color": 0x3498DB, "fields": fields}


def build_discord_failure_embed(
    *,
    topic: str | None,
    tone: str | None,
    news_item: NewsItem | None,
    error_message: str,
) -> dict[str, object]:
    fields: list[dict[str, object]] = [
        {"name": "Topic", "value": format_discord_field_value(topic), "inline": True},
        {"name": "Tone", "value": format_discord_field_value(tone), "inline": True},
    ]
    if news_item:
        fields.extend(
            [
                {
                    "name": "News title",
                    "value": format_discord_field_value(news_item.title),
                    "inline": False,
                },
                {
                    "name": "News source",
                    "value": format_discord_field_value(news_item.source),
                    "inline": True,
                },
                {
                    "name": "News published",
                    "value": format_news_published_at(news_item),
                    "inline": True,
                },
            ]
        )
    fields.append({"name": "Error", "value": error_message, "inline": False})
    return {"title": "Content bot failed", "color": 0xE74C3C, "fields": fields}


def send_discord_safely(
    config: AppConfig,
    embed: dict[str, object],
    *,
    warning_prefix: str,
) -> None:
    if not config.discord_notifications_enabled:
        return
    if not config.discord_webhook_url:
        print(f"{warning_prefix}: DISCORD_WEBHOOK_URL must be set.")
        return

    try:
        send_discord_embed(config, embed)
    except Exception as exc:
        print(f"{warning_prefix}: {exc}")


def send_discord_message_safely(
    config: AppConfig,
    message_text: str,
    *,
    warning_prefix: str,
) -> None:
    if not config.discord_notifications_enabled:
        return
    if not config.discord_webhook_url:
        print(f"{warning_prefix}: DISCORD_WEBHOOK_URL must be set.")
        return

    try:
        send_discord_message(config, message_text)
    except Exception as exc:
        print(f"{warning_prefix}: {exc}")


def send_failure_notifications(
    config: AppConfig,
    *,
    topic: str | None,
    tone: str | None,
    news_item: NewsItem | None,
    error_message: str,
) -> None:
    news_published_at = format_news_published_at(news_item) if news_item else None
    send_telegram_safely(
        config,
        build_failure_telegram_summary(
            topic=topic,
            tone=tone,
            error_message=error_message,
            news_title=news_item.title if news_item else None,
            news_source=news_item.source if news_item else None,
            news_published_at=news_published_at,
            news_url=news_item.link if news_item else None,
        ),
        warning_prefix="Warning: Telegram failure alert delivery failed",
    )
    send_discord_safely(
        config,
        build_discord_failure_embed(
            topic=topic,
            tone=tone,
            news_item=news_item,
            error_message=error_message,
        ),
        warning_prefix="Warning: Discord failure alert delivery failed",
    )


def send_success_notifications(
    config: AppConfig,
    *,
    topic: str,
    tone: str,
    final_post_text: str,
    elapsed: float,
    attempts: int,
    news_item: NewsItem | None,
    platform_results: list[PlatformLogResult] | None = None,
    partial: bool = False,
) -> None:
    news_published_at = format_news_published_at(news_item) if news_item else None
    send_telegram_safely(
        config,
        build_telegram_summary(
            topic=topic,
            tone=tone,
            tweet_text=final_post_text,
            time_taken_seconds=elapsed,
            attempts=attempts,
            news_title=news_item.title if news_item else None,
            news_source=news_item.source if news_item else None,
            news_published_at=news_published_at,
            news_url=news_item.link if news_item else None,
            platform_results=platform_results,
            partial=partial,
        ),
        warning_prefix="Warning: Telegram delivery failed",
    )
    send_discord_safely(
        config,
        build_discord_success_embed(
            topic=topic,
            tone=tone,
            tweet_text=final_post_text,
            time_taken_seconds=elapsed,
            attempts=attempts,
            news_item=news_item,
            platform_results=platform_results,
            partial=partial,
        ),
        warning_prefix="Warning: Discord delivery failed",
    )


def send_manual_notifications(
    config: AppConfig,
    *,
    topic: str,
    tone: str,
    final_post_text: str,
    elapsed: float,
    attempts: int,
    news_item: NewsItem | None,
) -> None:
    news_published_at = format_news_published_at(news_item) if news_item else None
    send_telegram_safely(
        config,
        build_telegram_summary(
            topic=topic,
            tone=tone,
            tweet_text=final_post_text,
            time_taken_seconds=elapsed,
            attempts=attempts,
            news_title=news_item.title if news_item else None,
            news_source=news_item.source if news_item else None,
            news_published_at=news_published_at,
            news_url=news_item.link if news_item else None,
        ),
        warning_prefix="Warning: Telegram delivery failed",
    )
    send_discord_safely(
        config,
        build_discord_manual_embed(
            topic=topic,
            tone=tone,
            time_taken_seconds=elapsed,
            attempts=attempts,
            news_item=news_item,
        ),
        warning_prefix="Warning: Discord delivery failed",
    )
    send_discord_message_safely(
        config,
        final_post_text,
        warning_prefix="Warning: Discord delivery failed",
    )
