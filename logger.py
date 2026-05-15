from __future__ import annotations

from datetime import datetime
from pathlib import Path

LOG_TITLE = "# Tweet History"


def build_slot_marker(*, run_date: str, run_slot: str) -> str:
    return f"<!-- tweet-slot:{run_date}:{run_slot} -->"


def has_logged_slot(log_file_path: Path, *, run_date: str, run_slot: str) -> bool:
    if not log_file_path.exists():
        return False
    marker = build_slot_marker(run_date=run_date, run_slot=run_slot)
    return marker in log_file_path.read_text(encoding="utf-8")


def build_tweet_log_entry(
    *,
    topic: str,
    tone: str,
    tweet_text: str,
    time_taken_seconds: float,
    attempts: int,
    tweet_url: str,
    run_slot: str | None = None,
    timestamp: str | None = None,
    run_date: str | None = None,
) -> str:
    resolved_timestamp = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    resolved_run_date = run_date or resolved_timestamp[:10]
    lines = [
        "",
        "## Tweet posted",
        "",
        f"- Date/time: {resolved_timestamp}",
    ]
    if run_slot:
        lines.append(f"- Run slot: {run_slot}")
    lines.extend(
        [
            f"- Topic: {topic}",
            f"- Tone: {tone}",
            f"- Time taken: {time_taken_seconds:.2f} seconds",
            f"- Attempts: {attempts}",
            f"- Tweet URL: {tweet_url}",
            "",
            "Tweet text:",
            "",
            f"> {tweet_text}",
        ]
    )
    if run_slot:
        lines.extend(
            [
                "",
                build_slot_marker(run_date=resolved_run_date, run_slot=run_slot),
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
) -> str:
    return "\n".join(
        [
            f"Topic: {topic}",
            f"Tone: {tone}",
            f"Time taken: {time_taken_seconds:.2f} seconds",
            f"Attempts: {attempts}",
            "",
            "Tweet text:",
            tweet_text,
        ]
    )


def append_tweet_log(
    log_file_path: Path,
    *,
    topic: str,
    tone: str,
    tweet_text: str,
    time_taken_seconds: float,
    attempts: int,
    tweet_url: str,
    run_slot: str | None = None,
    timestamp: str | None = None,
    run_date: str | None = None,
) -> None:
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    entry = build_tweet_log_entry(
        topic=topic,
        tone=tone,
        tweet_text=tweet_text,
        time_taken_seconds=time_taken_seconds,
        attempts=attempts,
        tweet_url=tweet_url,
        run_slot=run_slot,
        timestamp=timestamp,
        run_date=run_date,
    )
    append_log_entry(log_file_path, entry)


def append_log_entry(log_file_path: Path, entry: str) -> None:
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_file_path.exists():
        log_file_path.write_text(f"{LOG_TITLE}\n", encoding="utf-8")
    with log_file_path.open("a", encoding="utf-8") as log_file:
        log_file.write(entry)
