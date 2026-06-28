from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Literal
from urllib.parse import urlparse

import requests

from config import AppConfig
from discord_approval import is_authorized_approver, parse_discord_channel_id
from google_news_resolver import resolve_news_url
from link_preview import LinkMetadataParser, USER_AGENT, clean_card_text
from news_fetcher import NewsItem


OnDemandKind = Literal["direct_post", "news_url"]
URL_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)
TONE_PATTERN = re.compile(
    r"(?:^|\s)tone\s*=\s*(?:\"([^\"]+)\"|'([^']+)'|([^\n\r]+))",
    re.IGNORECASE,
)
NEWS_DATE_KEYS = (
    "article:published_time",
    "og:article:published_time",
    "date",
    "pubdate",
    "publishdate",
    "dc.date",
    "dc.date.issued",
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


@dataclass(frozen=True)
class OnDemandRequest:
    kind: OnDemandKind
    message_id: str
    author_id: str
    topic: str
    tone: str | None
    post_text: str | None = None
    news_item: NewsItem | None = None


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


def fetch_news_item_from_url(
    url: str,
    config: AppConfig,
    *,
    now: datetime | None = None,
) -> NewsItem:
    resolved_url = resolve_news_url(url, timeout_seconds=config.timeout_seconds)
    response = requests.get(
        resolved_url,
        timeout=min(config.timeout_seconds, 20),
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()

    final_url = response.url or resolved_url
    parser = LinkMetadataParser(final_url)
    parser.feed(response.text)
    raw_title = parser.title()
    if not raw_title:
        raise RuntimeError("Could not find a usable article title.")

    title = clean_card_text(raw_title, raw_title)
    description = clean_card_text(parser.description(), title)
    source = clean_card_text(
        parser.get("og:site_name", "application-name"),
        _source_from_url(final_url),
    )
    published_at = parse_news_published_at(
        parser.get(*NEWS_DATE_KEYS),
        now=now,
    )
    return NewsItem(
        title=title,
        source=source,
        published_at=published_at,
        link=final_url,
        summary=description,
    )


def _source_from_url(url: str) -> str:
    hostname = urlparse(url).hostname or "Unknown"
    if hostname.startswith("www."):
        hostname = hostname[4:]
    return hostname


def parse_news_published_at(
    value: str | None,
    *,
    now: datetime | None = None,
) -> datetime:
    if value:
        cleaned = value.strip()
        try:
            parsed = datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
        except ValueError:
            try:
                parsed = parsedate_to_datetime(cleaned)
            except (TypeError, ValueError):
                parsed = None
        if parsed is not None:
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)

    resolved_now = now or datetime.now(timezone.utc)
    if resolved_now.tzinfo is None:
        resolved_now = resolved_now.replace(tzinfo=timezone.utc)
    return resolved_now.astimezone(timezone.utc)


def fetch_next_on_demand_request(config: AppConfig) -> OnDemandRequest | None:
    return asyncio.run(_fetch_next_on_demand_request(config))


async def _fetch_next_on_demand_request(config: AppConfig) -> OnDemandRequest | None:
    try:
        import discord
    except ImportError as exc:
        raise RuntimeError("discord.py package is not installed.") from exc

    channel_id = parse_discord_channel_id(config.discord_channel_id)
    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)
    ready_event = asyncio.Event()
    selected_request: OnDemandRequest | None = None
    ready_error: Exception | None = None

    @client.event
    async def on_ready() -> None:
        nonlocal selected_request, ready_error
        try:
            channel = client.get_channel(channel_id)
            if channel is None:
                channel = await client.fetch_channel(channel_id)
            discord_messages = [
                message
                async for message in channel.history(
                    limit=config.on_demand_discord_lookback_limit
                )
            ]
            discord_messages.reverse()
            snapshots = [
                DiscordMessageSnapshot(
                    message_id=str(message.id),
                    author_id=str(message.author.id),
                    author_is_bot=bool(getattr(message.author, "bot", False)),
                    content=message.content or "",
                    referenced_message_id=(
                        str(message.reference.message_id)
                        if message.reference and message.reference.message_id
                        else None
                    ),
                )
                for message in discord_messages
            ]
            print(
                "Checked Discord on-demand requests: "
                f"{len(snapshots)} recent messages."
            )
            if snapshots and all(
                not snapshot.author_is_bot and not snapshot.content
                for snapshot in snapshots
            ):
                print(
                    "Warning: Discord message content was empty for recent messages. "
                    "Check the bot's Message Content Intent and channel permissions."
                )
            selection = select_on_demand_request(snapshots, config)
            if selection is None:
                print("No pending Discord on-demand request found.")
                return

            message_by_id = {str(message.id): message for message in discord_messages}
            source_message = message_by_id[selection.message_id]
            if selection.error:
                print(f"Discord on-demand request ignored: {selection.error}")
                await source_message.reply(selection.error, mention_author=False)
                return
            if selection.request is None:
                return

            try:
                selected_request = build_on_demand_request(
                    selection.message_id,
                    str(source_message.author.id),
                    selection.request,
                    config,
                )
            except Exception as exc:
                print(f"Discord on-demand request could not be used: {exc}")
                await source_message.reply(
                    f"Could not use on-demand request: {exc}",
                    mention_author=False,
                )
                return

            await source_message.reply(
                "Picked up this on-demand request for the next post run.",
                mention_author=False,
            )
        except Exception as exc:
            ready_error = exc
        finally:
            ready_event.set()

    client_task = asyncio.create_task(client.start(config.discord_bot_token or ""))
    ready_task = asyncio.create_task(ready_event.wait())
    try:
        done, _pending = await asyncio.wait(
            {client_task, ready_task},
            timeout=60,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if client_task in done:
            client_task.result()
        if ready_task not in done:
            raise RuntimeError("Discord on-demand bot did not become ready.")
        if ready_error is not None:
            raise RuntimeError(
                f"Discord on-demand setup failed: {ready_error}"
            ) from ready_error
    finally:
        ready_task.cancel()
        await client.close()
        try:
            await client_task
        except asyncio.CancelledError:
            pass
    return selected_request


def build_on_demand_request(
    message_id: str,
    author_id: str,
    parsed_request: ParsedOnDemandRequest,
    config: AppConfig,
) -> OnDemandRequest:
    if parsed_request.kind == "direct_post":
        return OnDemandRequest(
            kind="direct_post",
            message_id=message_id,
            author_id=author_id,
            topic="on-demand post",
            tone="direct",
            post_text=parsed_request.post_text,
        )

    if not parsed_request.news_url:
        raise RuntimeError("News URL is missing.")
    news_item = fetch_news_item_from_url(parsed_request.news_url, config)
    return OnDemandRequest(
        kind="news_url",
        message_id=message_id,
        author_id=author_id,
        topic="news",
        tone=parsed_request.tone,
        news_item=news_item,
    )
