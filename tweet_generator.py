#!/usr/bin/env python3
from __future__ import annotations

import random
import socket
import sys
import threading
import time
from datetime import timezone

from openai import OpenAIError

from bluesky_publisher import post_to_bluesky
from config import AppConfig, load_config
from discord_sender import send_discord_embed, send_discord_message
from generator import build_client, generate_valid_tweet
from logger import (
    append_log_entry,
    build_failure_telegram_summary,
    build_telegram_summary,
    build_tweet_log_entry,
)
from news_fetcher import NewsItem, fetch_latest_news
from publisher import build_post_text, max_generated_text_chars, post_tweet_to_x
from telegram_sender import send_telegram_message


def spinner(stop_event: threading.Event, message: str = "Generating tweet") -> None:
    frames = "|/-\\"
    idx = 0
    while not stop_event.is_set():
        frame = frames[idx % len(frames)]
        print(f"\r{message} {frame}", end="", flush=True)
        idx += 1
        time.sleep(0.1)
    clear_width = len(message) + 2
    print("\r" + (" " * clear_width) + "\r", end="", flush=True)


def stop_spinner(
    stop_event: threading.Event | None, spinner_thread: threading.Thread | None
) -> None:
    if spinner_thread and stop_event:
        stop_event.set()
        spinner_thread.join()


def format_timeout_message(config: AppConfig) -> str:
    return (
        "LLM request timed out after "
        f"{config.timeout_seconds} seconds while waiting for "
        f"{config.llm_model} at {config.llm_base_url}."
    )


def is_timeout_exception(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return True

    lowered = str(exc).lower()
    return "timed out" in lowered or "timeout" in lowered


def format_news_published_at(news_item: NewsItem) -> str:
    return news_item.published_at.astimezone(timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )


def describe_failure(exc: Exception, config: AppConfig | None = None) -> str:
    if isinstance(exc, OpenAIError):
        if looks_like_html_response(str(exc)):
            return (
                "LLM request failed: provider returned an HTML page. "
                "Check that LLM_BASE_URL points to an OpenAI-compatible API "
                "endpoint, not a website."
            )
        return f"LLM request failed: {exc}"
    if isinstance(exc, TimeoutError) and config:
        return format_timeout_message(config)
    if is_timeout_exception(exc) and config:
        return format_timeout_message(config)
    return str(exc) or exc.__class__.__name__


def looks_like_html_response(message: str) -> bool:
    lowered = message.lower()
    return "<!doctype html" in lowered or "<html" in lowered


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


def build_discord_success_embed(
    *,
    topic: str,
    tone: str,
    tweet_text: str,
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
    fields.append({"name": "Final tweet", "value": tweet_text, "inline": False})
    return {"title": "Tweet posted", "color": 0x2ECC71, "fields": fields}


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
    return {"title": "Tweet ready", "color": 0x3498DB, "fields": fields}


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
    return {"title": "Tweet bot failed", "color": 0xE74C3C, "fields": fields}


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
        build_discord_success_embed(
            topic=topic,
            tone=tone,
            tweet_text=final_post_text,
            time_taken_seconds=elapsed,
            attempts=attempts,
            news_item=news_item,
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


def run_once() -> int:
    process_start = time.perf_counter()
    try:
        config = load_config()
    except Exception as exc:
        print(f"Could not generate tweet: {exc}")
        return 0

    topic: str | None = None
    tone: str | None = None
    news_item: NewsItem | None = None
    stop_event: threading.Event | None = None
    spinner_thread: threading.Thread | None = None

    try:
        client = build_client(config)
        interactive_tty = sys.stdout.isatty()

        topic = random.choice(config.topics)
        tone = random.choice(config.tones)

        if config.news_enabled:
            try:
                news_item = fetch_latest_news(topic, config)
                if news_item is None:
                    print(
                        f"No recent RSS news found for {topic}. Using generic topic prompt."
                    )
                else:
                    print(f"Using RSS news: {news_item.title} ({news_item.source})")
            except Exception as exc:
                print(f"Warning: RSS news lookup failed for {topic}: {exc}")

        print("Generating tweet...")
        if interactive_tty:
            stop_event = threading.Event()
            spinner_thread = threading.Thread(
                target=spinner, args=(stop_event,), daemon=True
            )
            spinner_thread.start()

        news_url = news_item.link if news_item else None
        tweet, _generation_elapsed, attempts = generate_valid_tweet(
            client,
            config,
            topic,
            tone,
            news_item,
            max_tweet_chars=max_generated_text_chars(config.max_tweet_chars, news_url),
        )

        final_post_text = build_post_text(tweet, news_url)
        if config.post_to_bluesky:
            published = post_to_bluesky(config, final_post_text)
        elif config.post_to_x:
            published = post_tweet_to_x(config, tweet, news_url=news_url)
        else:
            stop_spinner(stop_event, spinner_thread)
            elapsed = time.perf_counter() - process_start
            send_manual_notifications(
                config,
                topic=topic,
                tone=tone,
                final_post_text=final_post_text,
                elapsed=elapsed,
                attempts=attempts,
                news_item=news_item,
            )
            print("Tweet ready for manual posting.")
            return 0

        stop_spinner(stop_event, spinner_thread)
        elapsed = time.perf_counter() - process_start
        log_entry = build_tweet_log_entry(
            topic=topic,
            tone=tone,
            tweet_text=final_post_text,
            time_taken_seconds=elapsed,
            attempts=attempts,
            tweet_url=published.url,
            news_title=news_item.title if news_item else None,
            news_source=news_item.source if news_item else None,
            news_published_at=format_news_published_at(news_item) if news_item else None,
            news_url=news_item.link if news_item else None,
        )
        append_log_entry(config.log_file_path, log_entry)
        send_success_notifications(
            config,
            topic=topic,
            tone=tone,
            final_post_text=final_post_text,
            elapsed=elapsed,
            attempts=attempts,
            news_item=news_item,
        )
        print("Tweet posted and logged.")
        return 0
    except Exception as exc:
        stop_spinner(stop_event, spinner_thread)
        error_message = describe_failure(exc, config)
        print(f"Could not complete tweet run: {error_message}")
        send_failure_notifications(
            config,
            topic=topic,
            tone=tone,
            news_item=news_item,
            error_message=error_message,
        )
        return 0


def main() -> int:
    return run_once()


if __name__ == "__main__":
    raise SystemExit(main())
