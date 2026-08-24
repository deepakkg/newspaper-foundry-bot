from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html import unescape
from urllib.parse import urlencode

import requests

from config import AppConfig
from google_news_resolver import resolve_news_url
from http_client import request_with_retry


GOOGLE_NEWS_RSS_SEARCH_URL = "https://news.google.com/rss/search"
USER_AGENT = "gemma-tweet-bot/1.0"


@dataclass(frozen=True)
class NewsItem:
    title: str
    source: str
    published_at: datetime
    link: str
    summary: str
    source_url: str | None = None


def build_google_news_rss_url(topic: str, *, language: str, region: str) -> str:
    normalized_language = language.strip().lower()
    normalized_region = region.strip().upper()
    query = urlencode(
        {
            "q": topic.strip(),
            "hl": f"{normalized_language}-{normalized_region}",
            "gl": normalized_region,
            "ceid": f"{normalized_region}:{normalized_language}",
        }
    )
    return f"{GOOGLE_NEWS_RSS_SEARCH_URL}?{query}"


def strip_html(text: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", text)
    return " ".join(unescape(without_tags).split())


def parse_published_at(value: str) -> datetime | None:
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def parse_rss_items(
    rss_text: str, *, now: datetime | None = None, recency_hours: int
) -> list[NewsItem]:
    resolved_now = now or datetime.now(timezone.utc)
    if resolved_now.tzinfo is None:
        resolved_now = resolved_now.replace(tzinfo=timezone.utc)
    resolved_now = resolved_now.astimezone(timezone.utc)
    recency_cutoff = resolved_now - timedelta(hours=recency_hours)

    root = ET.fromstring(rss_text)
    items: list[NewsItem] = []
    for item in root.findall("./channel/item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        description = strip_html(item.findtext("description") or "")
        published_at = parse_published_at(item.findtext("pubDate") or "")
        source_node = item.find("source")
        source = (
            source_node.text.strip()
            if source_node is not None and source_node.text
            else ""
        )

        if not title or not link or published_at is None:
            continue
        if published_at < recency_cutoff or published_at > resolved_now + timedelta(
            minutes=5
        ):
            continue

        items.append(
            NewsItem(
                title=title,
                source=source or "Google News",
                published_at=published_at,
                link=link,
                summary=description,
                source_url=link,
            )
        )

    return sorted(items, key=lambda news_item: news_item.published_at, reverse=True)


def _topic_tokens(topic: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", topic.lower()) if len(token) > 2}


def rank_news_items(
    items: list[NewsItem],
    topic: str,
    *,
    excluded_urls: set[str] | None = None,
    now: datetime | None = None,
) -> list[NewsItem]:
    excluded = {url.strip().lower() for url in (excluded_urls or set())}
    if excluded:
        available_items = [
            item
            for item in items
            if item.link.strip().lower() not in excluded
            and (item.source_url or "").strip().lower() not in excluded
        ]
        if not available_items:
            return []
        items = available_items
    resolved_now = now or datetime.now(timezone.utc)
    topic_tokens = _topic_tokens(topic)

    def score(item: NewsItem) -> tuple[float, float]:
        item_tokens = set(re.findall(r"[a-z0-9]+", f"{item.title} {item.summary}".lower()))
        relevance = len(topic_tokens & item_tokens)
        summary_bonus = 1 if item.summary and item.summary != item.title else 0
        age_hours = max((resolved_now - item.published_at).total_seconds() / 3600, 0)
        recency = max(0.0, 48.0 - age_hours) / 48.0
        return (relevance * 10 + summary_bonus * 2 + recency, -age_hours)

    return sorted(items, key=score, reverse=True)


def fetch_recent_news(
    topic: str,
    config: AppConfig,
    *,
    excluded_urls: set[str] | None = None,
) -> list[NewsItem]:
    url = build_google_news_rss_url(
        topic,
        language=config.news_language,
        region=config.news_region,
    )
    response = request_with_retry(
        requests.get,
        url,
        safe_to_retry=True,
        timeout=min(config.timeout_seconds, 20),
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    items = parse_rss_items(
        response.text,
        recency_hours=config.news_recency_hours,
    )
    return rank_news_items(items, topic, excluded_urls=excluded_urls)


def fetch_latest_news(
    topic: str,
    config: AppConfig,
    *,
    excluded_urls: set[str] | None = None,
) -> NewsItem | None:
    items = fetch_recent_news(topic, config, excluded_urls=excluded_urls)
    if not items:
        return None

    latest_item = items[0]
    resolved_link = resolve_news_url(
        latest_item.link,
        timeout_seconds=config.timeout_seconds,
    )
    if resolved_link == latest_item.link:
        return latest_item
    return replace(latest_item, link=resolved_link)
