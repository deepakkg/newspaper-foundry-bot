from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from config import AppConfig
from discord_approval import is_authorized_approver


OnDemandKind = Literal["direct_post", "news_url"]
URL_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)
TONE_PATTERN = re.compile(
    r"(?:^|\s)tone\s*=\s*(?:\"([^\"]+)\"|'([^']+)'|([^\n\r]+))",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DiscordMessageSnapshot:
    message_id: str
    author_id: str
    author_is_bot: bool
    content: str
    referenced_message_id: str | None = None


@dataclass(frozen=True)
class ParsedOnDemandRequest:
    kind: OnDemandKind
    post_text: str | None = None
    news_url: str | None = None
    tone: str | None = None


@dataclass(frozen=True)
class SelectedOnDemandRequest:
    message_id: str
    request: ParsedOnDemandRequest | None = None
    error: str | None = None


def command_kind(content: str) -> OnDemandKind | None:
    stripped = content.lstrip()
    lowered = stripped.lower()
    if lowered == "/post" or lowered.startswith(("/post ", "/post\n", "/post\r")):
        return "direct_post"
    if lowered == "/news" or lowered.startswith(("/news ", "/news\n", "/news\r")):
        return "news_url"
    if starts_with_url(stripped):
        return "news_url"
    return None


def starts_with_url(content: str) -> bool:
    match = URL_PATTERN.match(content)
    return bool(match)


def parse_on_demand_command(content: str, config: AppConfig) -> ParsedOnDemandRequest:
    kind = command_kind(content)
    if kind is None:
        raise ValueError("Unsupported on-demand command.")

    stripped = content.lstrip()
    command_parts = stripped.split(maxsplit=1)
    lowered = stripped.lower()
    if lowered == "/post" or lowered.startswith(("/post ", "/post\n", "/post\r")):
        body = command_parts[1].strip() if len(command_parts) > 1 else ""
    elif lowered == "/news" or lowered.startswith(("/news ", "/news\n", "/news\r")):
        body = command_parts[1].strip() if len(command_parts) > 1 else ""
    else:
        body = stripped
    if kind == "direct_post" and not body and "\n" in stripped:
        body = stripped.split("\n", 1)[1].strip()

    if kind == "direct_post":
        if not body:
            raise ValueError("Direct post text is missing.")
        return ParsedOnDemandRequest(kind=kind, post_text=body)

    url_match = URL_PATTERN.search(body)
    if not url_match:
        raise ValueError("News URL is missing.")
    news_url = url_match.group(0).rstrip(".,)>]\"'")
    return ParsedOnDemandRequest(
        kind=kind,
        news_url=news_url,
        tone=parse_requested_tone(body, config.tones),
    )


def parse_requested_tone(content: str, configured_tones: list[str]) -> str | None:
    match = TONE_PATTERN.search(content)
    if not match:
        return None
    raw_tone = next((group for group in match.groups() if group), "").strip()
    if not raw_tone:
        return None
    normalized_requested = " ".join(raw_tone.split()).lower()
    for tone in configured_tones:
        if tone.lower() == normalized_requested:
            return tone
    return None


def select_on_demand_request(
    messages: list[DiscordMessageSnapshot],
    config: AppConfig,
) -> SelectedOnDemandRequest | None:
    processed_message_ids = {
        message.referenced_message_id
        for message in messages
        if message.author_is_bot and message.referenced_message_id
    }
    candidates = [
        message
        for message in messages
        if not message.author_is_bot
        and message.message_id not in processed_message_ids
        and command_kind(message.content) is not None
    ]
    if not candidates:
        return None

    selected = (
        _first_by_kind(candidates, "direct_post")
        or _first_explicit_news_command(candidates)
        or _first_plain_url(candidates)
    )
    if selected is None:
        return None

    if not is_authorized_approver(config, selected.author_id):
        return SelectedOnDemandRequest(
            message_id=selected.message_id,
            error="You are not allowed to submit on-demand content requests.",
        )

    try:
        request = parse_on_demand_command(selected.content, config)
    except ValueError as exc:
        return SelectedOnDemandRequest(message_id=selected.message_id, error=str(exc))
    return SelectedOnDemandRequest(message_id=selected.message_id, request=request)


def _first_by_kind(
    messages: list[DiscordMessageSnapshot],
    kind: OnDemandKind,
) -> DiscordMessageSnapshot | None:
    for message in messages:
        if command_kind(message.content) == kind:
            return message
    return None


def _first_explicit_news_command(
    messages: list[DiscordMessageSnapshot],
) -> DiscordMessageSnapshot | None:
    for message in messages:
        stripped = message.content.lstrip().lower()
        if stripped == "/news" or stripped.startswith(("/news ", "/news\n", "/news\r")):
            return message
    return None


def _first_plain_url(
    messages: list[DiscordMessageSnapshot],
) -> DiscordMessageSnapshot | None:
    for message in messages:
        if starts_with_url(message.content.lstrip()):
            return message
    return None
