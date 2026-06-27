#!/usr/bin/env python3
from __future__ import annotations

import random
import socket
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAIError

from article_links import build_article_link_entry, update_article_links_page
from bluesky_publisher import post_to_bluesky
from cloudinary_uploader import upload_image_to_cloudinary
from config import AppConfig, load_config
from discord_approval import (
    ApprovalRequest,
    request_discord_approval,
)
from generator import build_client, generate_valid_tweet
from instagram_content import build_instagram_caption, generate_instagram_hashtags
from instagram_image import render_instagram_image
from instagram_publisher import publish_instagram_image
from logger import PlatformLogResult, append_log_entry, build_run_log_entry
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


@dataclass(frozen=True)
class PublishOutcome:
    results: list[PlatformLogResult]
    success_count: int


def enabled_platforms(config: AppConfig) -> list[str]:
    platforms: list[str] = []
    if config.post_to_bluesky:
        platforms.append("Bluesky")
    if config.post_to_instagram:
        platforms.append("Instagram")
    if config.post_to_x:
        platforms.append("X")
    return platforms


def image_output_path(config: AppConfig, topic: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = "-".join(part for part in topic.lower().split() if part) or "post"
    safe_slug = "".join(char for char in slug if char.isalnum() or char == "-")[:60]
    return config.generated_image_dir / f"{timestamp}-{safe_slug}.png"


def format_platform_result(result: PlatformLogResult) -> str:
    detail_parts = [result.status]
    if result.url:
        detail_parts.append(result.url)
    if result.identifier:
        detail_parts.append(result.identifier)
    if result.error:
        detail_parts.append(result.error)
    return f"{result.platform}: {' | '.join(detail_parts)}"


def print_platform_results(results: list[PlatformLogResult]) -> None:
    for result in results:
        print(format_platform_result(result))


def update_article_links_after_instagram_publish(
    config: AppConfig,
    *,
    news_item: NewsItem | None,
    results: list[PlatformLogResult],
) -> None:
    if not config.article_links_enabled or news_item is None:
        return

    instagram_result = next(
        (
            result
            for result in results
            if result.platform == "Instagram" and result.status == "published"
        ),
        None,
    )
    if instagram_result is None:
        return

    try:
        update_article_links_page(
            config,
            build_article_link_entry(
                news_item,
                instagram_media_id=instagram_result.identifier,
                instagram_url=instagram_result.url,
            ),
        )
        print(f"Article link page updated: {news_item.link}")
    except Exception as exc:
        print(f"Warning: Article link page update failed: {exc}")


def publish_enabled_platforms(
    config: AppConfig,
    *,
    topic: str,
    tweet: str,
    final_post_text: str,
    news_item: NewsItem | None,
    instagram_caption: str | None,
) -> PublishOutcome:
    news_url = news_item.link if news_item else None
    results: list[PlatformLogResult] = []
    success_count = 0

    if config.post_to_bluesky:
        try:
            published = post_to_bluesky(
                config,
                build_post_text_without_url(tweet),
                news_url=news_url,
                news_title=news_item.title if news_item else None,
                news_summary=news_item.summary if news_item else None,
            )
            success_count += 1
            results.append(
                PlatformLogResult(
                    platform="Bluesky",
                    status="published",
                    url=published.url,
                    identifier=published.uri,
                )
            )
        except Exception as exc:
            results.append(
                PlatformLogResult(platform="Bluesky", status="failed", error=str(exc))
            )

    if config.post_to_instagram:
        cloudinary_url: str | None = None
        try:
            if instagram_caption is None:
                raise RuntimeError("Instagram caption was not generated.")
            image_path = render_instagram_image(
                tweet,
                image_output_path(config, topic),
            )
            uploaded = upload_image_to_cloudinary(config, image_path)
            cloudinary_url = uploaded.secure_url
            published = publish_instagram_image(
                config,
                image_url=uploaded.secure_url,
                caption=instagram_caption,
            )
            success_count += 1
            results.append(
                PlatformLogResult(
                    platform="Instagram",
                    status="published",
                    url=published.url,
                    identifier=published.media_id,
                )
            )
        except Exception as exc:
            error_message = str(exc)
            if cloudinary_url:
                error_message = (
                    f"{error_message}; Cloudinary URL: {cloudinary_url}"
                )
            results.append(
                PlatformLogResult(
                    platform="Instagram",
                    status="failed",
                    error=error_message,
                )
            )

    if config.post_to_x:
        try:
            published = post_tweet_to_x(config, tweet, news_url=news_url)
            success_count += 1
            results.append(
                PlatformLogResult(
                    platform="X",
                    status="published",
                    url=published.url,
                    identifier=published.tweet_id,
                )
            )
        except Exception as exc:
            results.append(PlatformLogResult(platform="X", status="failed", error=str(exc)))

    return PublishOutcome(results=results, success_count=success_count)


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

        print("Generating post...")
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
        target_platforms = enabled_platforms(config)
        instagram_caption = None
        if config.post_to_instagram:
            instagram_caption = build_instagram_caption(
                topic=topic,
                tone=tone,
                news_item=news_item,
                llm_hashtags=generate_instagram_hashtags(
                    client,
                    config,
                    topic,
                    tone,
                    news_item,
                ),
                article_link_in_bio=config.article_links_enabled,
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
