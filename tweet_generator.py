#!/usr/bin/env python3
from __future__ import annotations

import random
import socket
import sys
import threading
import time
from datetime import timezone

from ollama import ResponseError

from config import AppConfig, load_config
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
        "Ollama request timed out after "
        f"{config.timeout_seconds} seconds while waiting for "
        f"{config.ollama_model} at {config.ollama_host}."
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
    if isinstance(exc, ResponseError):
        return f"Ollama request failed: {exc}"
    if isinstance(exc, TimeoutError) and config:
        return format_timeout_message(config)
    if is_timeout_exception(exc) and config:
        return format_timeout_message(config)
    return str(exc) or exc.__class__.__name__


def send_failure_telegram(
    config: AppConfig,
    *,
    topic: str | None,
    tone: str | None,
    news_item: NewsItem | None,
    error_message: str,
) -> None:
    if not config.telegram_bot_token or not config.telegram_chat_id:
        return

    try:
        send_telegram_message(
            config,
            build_failure_telegram_summary(
                topic=topic,
                tone=tone,
                error_message=error_message,
                news_title=news_item.title if news_item else None,
                news_source=news_item.source if news_item else None,
                news_published_at=(
                    format_news_published_at(news_item) if news_item else None
                ),
                news_url=news_item.link if news_item else None,
            ),
        )
    except Exception as exc:
        print(f"Warning: Telegram failure alert delivery failed: {exc}")


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

        if not config.post_to_x:
            raise RuntimeError("POST_TO_X is disabled. Enable it to post and log tweets.")

        final_post_text = build_post_text(tweet, news_url)
        published = post_tweet_to_x(config, tweet, news_url=news_url)
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
        if config.telegram_bot_token and config.telegram_chat_id:
            try:
                send_telegram_message(
                    config,
                    build_telegram_summary(
                        topic=topic,
                        tone=tone,
                        tweet_text=final_post_text,
                        time_taken_seconds=elapsed,
                        attempts=attempts,
                        news_title=news_item.title if news_item else None,
                        news_source=news_item.source if news_item else None,
                        news_published_at=(
                            format_news_published_at(news_item) if news_item else None
                        ),
                        news_url=news_item.link if news_item else None,
                    ),
                )
            except Exception as exc:
                print(f"Warning: Telegram delivery failed: {exc}")
        print("Tweet posted and logged.")
        return 0
    except Exception as exc:
        stop_spinner(stop_event, spinner_thread)
        error_message = describe_failure(exc, config)
        print(f"Could not complete tweet run: {error_message}")
        send_failure_telegram(
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
