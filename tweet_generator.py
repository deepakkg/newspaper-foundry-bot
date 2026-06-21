#!/usr/bin/env python3
from __future__ import annotations

import random
import socket
import sys
import threading
import time

from openai import OpenAIError

from bluesky_publisher import post_to_bluesky
from config import AppConfig, load_config
from generator import build_client, generate_valid_tweet
from logger import append_log_entry, build_tweet_log_entry
from news_fetcher import NewsItem, fetch_latest_news
from notifications import (
    format_news_published_at,
    send_failure_notifications,
    send_manual_notifications,
    send_success_notifications,
)
from publisher import (
    build_post_text,
    build_post_text_without_url,
    max_generated_text_chars,
    post_tweet_to_x,
)


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
            published = post_to_bluesky(
                config,
                build_post_text_without_url(tweet),
                news_url=news_url,
                news_title=news_item.title if news_item else None,
                news_summary=news_item.summary if news_item else None,
            )
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
