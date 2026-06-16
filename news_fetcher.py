from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html import unescape
from urllib.parse import urlencode, urlparse

import requests

from config import AppConfig


GOOGLE_NEWS_RSS_SEARCH_URL = "https://news.google.com/rss/search"
USER_AGENT = "gemma-tweet-bot/1.0"


@dataclass(frozen=True)
class NewsItem:
    title: str
    source: str
    published_at: datetime
    link: str
    summary: str


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


def is_google_url(url: str) -> bool:
    hostname = (urlparse(url).hostname or "").lower()
    return hostname == "google.com" or hostname.endswith(".google.com")


def resolve_news_url(url: str, *, timeout_seconds: int) -> str:
    cleaned_url = url.strip()
    if not cleaned_url or not is_google_url(cleaned_url):
        return cleaned_url

    try:
        response = requests.get(
            cleaned_url,
            timeout=min(timeout_seconds, 10),
            headers={"User-Agent": USER_AGENT},
            allow_redirects=True,
        )
        response.raise_for_status()
    except requests.RequestException:
        return cleaned_url

    resolved_url = response.url.strip()
    if not resolved_url or is_google_url(resolved_url):
        return cleaned_url
    return resolved_url


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
        source = source_node.text.strip() if source_node is not None and source_node.text else ""

        if not title or not link or published_at is None:
            continue
        if published_at < recency_cutoff or published_at > resolved_now + timedelta(minutes=5):
            continue

        items.append(
            NewsItem(
                title=title,
                source=source or "Google News",
                published_at=published_at,
                link=link,
                summary=description,
            )
        )

    return sorted(items, key=lambda news_item: news_item.published_at, reverse=True)


def fetch_latest_news(topic: str, config: AppConfig) -> NewsItem | None:
    url = build_google_news_rss_url(
        topic,
        language=config.news_language,
        region=config.news_region,
    )
    response = requests.get(
        url,
        timeout=min(config.timeout_seconds, 20),
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    items = parse_rss_items(
        response.text,
        recency_hours=config.news_recency_hours,
    )
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
