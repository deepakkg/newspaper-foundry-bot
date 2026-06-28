#!/usr/bin/env python3
from __future__ import annotations

import socket
import sys
import threading
import time

from openai import OpenAIError

from config import AppConfig, load_config
from content_source import select_content_source
from discord_approval import (
    ApprovalRequest,
    request_discord_approval,
)
from generator import build_client, generate_valid_tweet
from instagram_content import (
    build_instagram_caption,
    generate_instagram_hashtags,
    generate_instagram_hashtags_from_text,
)
from logger import PlatformLogResult, append_log_entry, build_run_log_entry
from news_fetcher import NewsItem
from notifications import (
    format_news_published_at,
    send_failure_notifications,
    send_manual_notifications,
    send_success_notifications,
)
from publishing_flow import (
    enabled_platforms,
    print_platform_results,
    publish_enabled_platforms,
    update_article_links_after_instagram_publish,
)
from publisher import (
    build_post_text,
    max_generated_text_chars,
)


def spinner(stop_event: threading.Event, message: str = "Generating post") -> None:
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
        print(f"Could not generate post: {exc}")
        return 0

    topic: str | None = None
    tone: str | None = None
    news_item: NewsItem | None = None
    stop_event: threading.Event | None = None
    spinner_thread: threading.Thread | None = None

    try:
        client = build_client(config)
        interactive_tty = sys.stdout.isatty()

        content_source = select_content_source(config)
        topic = content_source.topic
        tone = content_source.tone
        news_item = content_source.news_item
        on_demand_request = content_source.on_demand_request

        print("Generating post...")
        if interactive_tty:
            stop_event = threading.Event()
            spinner_thread = threading.Thread(
                target=spinner, args=(stop_event,), daemon=True
            )
            spinner_thread.start()

        news_url = news_item.link if news_item else None
        if on_demand_request and on_demand_request.kind == "direct_post":
            tweet = on_demand_request.post_text or ""
            attempts = 0
        else:
            tweet, _generation_elapsed, attempts = generate_valid_tweet(
                client,
                config,
                topic,
                tone,
                news_item,
                max_tweet_chars=max_generated_text_chars(config.max_tweet_chars, news_url),
            )

        final_post_text = build_post_text(tweet, news_url)
        target_platforms = enabled_platforms(config)
        instagram_caption = None
        if config.post_to_instagram:
            direct_post_request = (
                on_demand_request is not None
                and on_demand_request.kind == "direct_post"
            )
            if direct_post_request:
                llm_hashtags = generate_instagram_hashtags_from_text(
                    client,
                    config,
                    tweet,
                )
            else:
                llm_hashtags = generate_instagram_hashtags(
                    client,
                    config,
                    topic,
                    tone,
                    news_item,
                )
            instagram_caption = build_instagram_caption(
                topic=topic,
                tone=tone,
                news_item=news_item,
                llm_hashtags=llm_hashtags,
                article_link_in_bio=config.article_links_enabled,
                include_topic_tone_hashtags=not direct_post_request,
            )

        if not target_platforms:
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
            print("Post ready for manual publishing.")
            return 0

        stop_spinner(stop_event, spinner_thread)
        elapsed = time.perf_counter() - process_start
        decision_by = None
        if config.approval_required:
            approval = request_discord_approval(
                config,
                ApprovalRequest(
                    topic=topic,
                    tone=tone,
                    final_post_text=final_post_text,
                    instagram_caption=instagram_caption,
                    elapsed=elapsed,
                    attempts=attempts,
                    target_platforms=target_platforms,
                    news_item=news_item,
                ),
            )
            decision_by = approval.username or approval.user_id

            if approval.status in {"declined", "expired"}:
                title = "Post declined" if approval.status == "declined" else "Post expired"
                append_log_entry(
                    config.log_file_path,
                    build_run_log_entry(
                        title=title,
                        topic=topic,
                        tone=tone,
                        post_text=final_post_text,
                        time_taken_seconds=elapsed,
                        attempts=attempts,
                        platform_results=[
                            PlatformLogResult(
                                platform=platform,
                                status="not published",
                                error=approval.status,
                            )
                            for platform in target_platforms
                        ],
                        news_title=news_item.title if news_item else None,
                        news_source=news_item.source if news_item else None,
                        news_published_at=format_news_published_at(news_item)
                        if news_item
                        else None,
                        news_url=news_item.link if news_item else None,
                        instagram_caption=instagram_caption,
                        decision_by=decision_by,
                    ),
                )
                print("Post was not published.")
                return 0

        outcome = publish_enabled_platforms(
            config,
            topic=topic,
            tweet=tweet,
            final_post_text=final_post_text,
            news_item=news_item,
            instagram_caption=instagram_caption,
        )
        print_platform_results(outcome.results)
        update_article_links_after_instagram_publish(
            config,
            news_item=news_item,
            results=outcome.results,
        )

        if outcome.success_count == 0:
            append_log_entry(
                config.log_file_path,
                build_run_log_entry(
                    title="Post publish failed",
                    topic=topic,
                    tone=tone,
                    post_text=final_post_text,
                    time_taken_seconds=elapsed,
                    attempts=attempts,
                    platform_results=outcome.results,
                    news_title=news_item.title if news_item else None,
                    news_source=news_item.source if news_item else None,
                    news_published_at=format_news_published_at(news_item)
                    if news_item
                    else None,
                    news_url=news_item.link if news_item else None,
                    instagram_caption=instagram_caption,
                    decision_by=decision_by,
                ),
            )
            errors = "; ".join(
                f"{result.platform}: {result.error}" for result in outcome.results
            )
            raise RuntimeError(f"All enabled platforms failed to publish. {errors}")

        partial = any(result.status == "failed" for result in outcome.results)
        log_entry = build_run_log_entry(
            title="Post partially published" if partial else "Post published",
            topic=topic,
            tone=tone,
            post_text=final_post_text,
            time_taken_seconds=elapsed,
            attempts=attempts,
            platform_results=outcome.results,
            news_title=news_item.title if news_item else None,
            news_source=news_item.source if news_item else None,
            news_published_at=format_news_published_at(news_item) if news_item else None,
            news_url=news_item.link if news_item else None,
            instagram_caption=instagram_caption,
            decision_by=decision_by,
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
            platform_results=outcome.results,
            partial=partial,
        )
        if partial:
            print("Post partially published and logged.")
        else:
            print("Post published and logged.")
        return 0
    except Exception as exc:
        stop_spinner(stop_event, spinner_thread)
        error_message = describe_failure(exc, config)
        print(f"Could not complete post run: {error_message}")
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
