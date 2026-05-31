#!/usr/bin/env python3
from __future__ import annotations

import random
import socket
import sys
import threading
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from ollama import ResponseError

from config import AppConfig, load_config
from generator import build_client, generate_valid_tweet
from logger import append_log_entry, build_telegram_summary, build_tweet_log_entry
from news_fetcher import NewsItem, fetch_latest_news
from publisher import post_tweet_to_x
from schedule_guard import RunDecision, decide_scheduled_run
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


def run_once(
    *,
    respect_schedule: bool = False,
    force_run: bool = False,
    now: datetime | None = None,
) -> int:
    process_start = time.perf_counter()
    try:
        config = load_config()
    except Exception as exc:
        print(f"Could not generate tweet: {exc}")
        return 1

    run_decision: RunDecision | None = None
    if respect_schedule and not force_run:
        run_decision = decide_scheduled_run(config, now)
        if not run_decision.should_run:
            print(run_decision.reason)
            return 0
        print(run_decision.reason)
    elif respect_schedule and force_run:
        timezone = ZoneInfo(config.run_timezone)
        resolved_now = now.astimezone(timezone) if now else datetime.now(timezone)
        run_decision = RunDecision(
            should_run=True,
            run_date=resolved_now.date().isoformat(),
            run_slot=resolved_now.strftime("%H:%M"),
            reason="Forced run requested.",
        )
        print(run_decision.reason)

    client = build_client(config)
    interactive_tty = sys.stdout.isatty()

    topic = random.choice(config.topics)
    tone = random.choice(config.tones)
    news_item: NewsItem | None = None

    if config.news_enabled:
        try:
            news_item = fetch_latest_news(topic, config)
            if news_item is None:
                print(f"No recent RSS news found for {topic}. Using generic topic prompt.")
            else:
                print(f"Using RSS news: {news_item.title} ({news_item.source})")
        except Exception as exc:
            print(f"Warning: RSS news lookup failed for {topic}: {exc}")

    stop_event: threading.Event | None = None
    spinner_thread: threading.Thread | None = None

    try:
        print("Generating tweet...")
        if interactive_tty:
            stop_event = threading.Event()
            spinner_thread = threading.Thread(
                target=spinner, args=(stop_event,), daemon=True
            )
            spinner_thread.start()

        tweet, _generation_elapsed, attempts = generate_valid_tweet(
            client,
            config,
            topic,
            tone,
            news_item,
        )

        if not config.post_to_x:
            raise RuntimeError("POST_TO_X is disabled. Enable it to post and log tweets.")

        published = post_tweet_to_x(config, tweet)
        stop_spinner(stop_event, spinner_thread)
        elapsed = time.perf_counter() - process_start
        timestamp = None
        run_date = None
        run_slot = None
        if run_decision:
            timezone = ZoneInfo(config.run_timezone)
            resolved_now = now.astimezone(timezone) if now else datetime.now(timezone)
            timestamp = resolved_now.strftime("%Y-%m-%d %H:%M:%S %Z")
            run_date = run_decision.run_date
            run_slot = run_decision.run_slot
        log_entry = build_tweet_log_entry(
            topic=topic,
            tone=tone,
            tweet_text=tweet,
            time_taken_seconds=elapsed,
            attempts=attempts,
            tweet_url=published.url,
            run_slot=run_slot,
            timestamp=timestamp,
            run_date=run_date,
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
                        tweet_text=tweet,
                        time_taken_seconds=elapsed,
                        attempts=attempts,
                    ),
                )
            except Exception as exc:
                print(f"Warning: Telegram delivery failed: {exc}")
        print("Tweet posted and logged.")
        return 0
    except ResponseError as exc:
        stop_spinner(stop_event, spinner_thread)
        print(f"Ollama request failed: {exc}")
        return 1
    except TimeoutError:
        stop_spinner(stop_event, spinner_thread)
        print(format_timeout_message(config))
        return 1
    except Exception as exc:
        stop_spinner(stop_event, spinner_thread)
        if is_timeout_exception(exc):
            print(format_timeout_message(config))
            return 1
        print(f"Could not generate tweet: {exc}")
        return 1


def main() -> int:
    respect_schedule = "--respect-schedule" in sys.argv[1:]
    force_run = "--force" in sys.argv[1:]
    return run_once(respect_schedule=respect_schedule, force_run=force_run)


if __name__ == "__main__":
    raise SystemExit(main())
