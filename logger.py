from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

LOG_TITLE = "# Post History"


@dataclass(frozen=True)
class PlatformLogResult:
    platform: str
    status: str
    url: str | None = None
    identifier: str | None = None
    error: str | None = None


def build_tweet_log_entry(
    *,
    topic: str,
    tone: str,
    tweet_text: str,
    time_taken_seconds: float,
    attempts: int,
    tweet_url: str,
    timestamp: str | None = None,
    news_title: str | None = None,
    news_source: str | None = None,
    news_published_at: str | None = None,
    news_url: str | None = None,
) -> str:
    resolved_timestamp = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "",
        "## Post published",
        "",
        f"- Date/time: {resolved_timestamp}",
    ]
    lines.extend(
        [
            f"- Topic: {topic}",
            f"- Tone: {tone}",
            f"- Time taken: {time_taken_seconds:.2f} seconds",
            f"- Attempts: {attempts}",
            f"- Post URL: {tweet_url}",
        ]
    )
    if news_title:
        lines.extend(
            [
                f"- News title: {news_title}",
                f"- News source: {news_source or 'Unknown'}",
            ]
        )
        if news_published_at:
            lines.append(f"- News published: {news_published_at}")
        if news_url:
            lines.append(f"- News URL: {news_url}")
    lines.extend(
        [
            "",
            "Post text:",
            "",
            f"> {tweet_text}",
        ]
    )
    lines.append("")
    return "\n".join(lines)


def build_telegram_summary(
    *,
    topic: str,
    tone: str,
    tweet_text: str,
    time_taken_seconds: float,
    attempts: int,
    news_title: str | None = None,
    news_source: str | None = None,
    news_url: str | None = None,
    news_published_at: str | None = None,
) -> str:
    lines = [
        f"Topic: {topic}",
        f"Tone: {tone}",
        f"Time taken: {time_taken_seconds:.2f} seconds",
        f"Attempts: {attempts}",
    ]
    if news_title:
        lines.extend(
            [
                "",
                "News reference:",
                f"{news_title} ({news_source or 'Unknown'})",
            ]
        )
        if news_published_at:
            lines.append(f"Published: {news_published_at}")
    lines.extend(
        [
            "",
            "Post text:",
            tweet_text,
        ]
    )
    return "\n".join(lines)


def build_failure_telegram_summary(
    *,
    error_message: str,
    topic: str | None = None,
    tone: str | None = None,
    news_title: str | None = None,
    news_source: str | None = None,
    news_url: str | None = None,
    news_published_at: str | None = None,
) -> str:
    lines = [
        "Content bot failed",
        f"Topic: {topic or 'Not selected'}",
        f"Tone: {tone or 'Not selected'}",
    ]
    if news_title:
        lines.extend(
            [
                "",
                "News reference:",
                f"{news_title} ({news_source or 'Unknown'})",
            ]
        )
        if news_published_at:
            lines.append(f"Published: {news_published_at}")
        if news_url:
            lines.append(news_url)
    lines.extend(
        [
            "",
            "Error:",
            error_message,
        ]
    )
    return "\n".join(lines)


def append_log_entry(log_file_path: Path, entry: str) -> None:
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_file_path.exists():
        log_file_path.write_text(f"{LOG_TITLE}\n", encoding="utf-8")
    with log_file_path.open("a", encoding="utf-8") as log_file:
        log_file.write(entry)


def build_run_log_entry(
    *,
    title: str,
    topic: str,
    tone: str,
    post_text: str,
    time_taken_seconds: float,
    attempts: int,
    platform_results: list[PlatformLogResult],
    timestamp: str | None = None,
    news_title: str | None = None,
    news_source: str | None = None,
    news_published_at: str | None = None,
    news_url: str | None = None,
    instagram_caption: str | None = None,
    decision_by: str | None = None,
) -> str:
    resolved_timestamp = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "",
        f"## {title}",
        "",
        f"- Date/time: {resolved_timestamp}",
        f"- Topic: {topic}",
        f"- Tone: {tone}",
        f"- Time taken: {time_taken_seconds:.2f} seconds",
        f"- Attempts: {attempts}",
    ]
    if decision_by:
        lines.append(f"- Decision by: {decision_by}")
    if news_title:
        lines.extend(
            [
                f"- News title: {news_title}",
                f"- News source: {news_source or 'Unknown'}",
            ]
        )
        if news_published_at:
            lines.append(f"- News published: {news_published_at}")
        if news_url:
            lines.append(f"- News URL: {news_url}")
    if platform_results:
        lines.extend(["", "Platform results:"])
        for result in platform_results:
            detail_parts = [result.status]
            if result.url:
                detail_parts.append(result.url)
            if result.identifier:
                detail_parts.append(result.identifier)
            if result.error:
                detail_parts.append(result.error)
            lines.append(f"- {result.platform}: {' | '.join(detail_parts)}")
    lines.extend(["", "Post text:", "", f"> {post_text}", ""])
    if instagram_caption:
        lines.extend(["Instagram caption:", "", instagram_caption, ""])
    return "\n".join(lines)
