from __future__ import annotations

import base64
import binascii
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html import unescape
from urllib.parse import quote, urlencode, urlparse

import requests

from config import AppConfig


GOOGLE_NEWS_RSS_SEARCH_URL = "https://news.google.com/rss/search"
GOOGLE_NEWS_BATCH_EXECUTE_URL = (
    "https://news.google.com/_/DotsSplashUi/data/batchexecute"
)
GOOGLE_NEWS_DECODE_RPC_ID = "Fbv4je"
USER_AGENT = "gemma-tweet-bot/1.0"
GOOGLE_NEWS_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
)
URL_PATTERN = re.compile(r"https?://[^\s\"'<>\\]+")
GOOGLE_NEWS_ID_PATTERN = re.compile(r"AU_[A-Za-z0-9_-]+")
GOOGLE_NEWS_SIGNATURE_PATTERN = re.compile(r'data-n-a-sg="([^"]+)"')
GOOGLE_NEWS_TIMESTAMP_PATTERN = re.compile(r'data-n-a-ts="([^"]+)"')


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


def is_publisher_url(url: str) -> bool:
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False
    return not is_google_url(url)


def extract_google_news_token(url: str) -> str | None:
    parsed = urlparse(url)
    parts = [part for part in parsed.path.split("/") if part]
    for marker in ("articles", "read"):
        if marker in parts:
            marker_index = parts.index(marker)
            if marker_index + 1 < len(parts):
                return parts[marker_index + 1]
    return None


def extract_first_publisher_url(text: str) -> str | None:
    variants = {
        text,
        unescape(text),
        text.replace("\\/", "/"),
        text.replace("\\\\/", "/"),
        unescape(text).replace("\\/", "/"),
        unescape(text).replace("\\\\/", "/"),
    }

    for variant in variants:
        for match in URL_PATTERN.finditer(variant):
            candidate = match.group(0).rstrip(".,)]}")
            if is_publisher_url(candidate):
                return candidate
    return None


def decode_embedded_news_url(token: str) -> str | None:
    try:
        decoded = base64.urlsafe_b64decode(token + ("=" * (-len(token) % 4)))
    except (ValueError, binascii.Error):
        return None

    decoded_text = decoded.decode("utf-8", errors="ignore")
    return extract_first_publisher_url(decoded_text)


def decode_google_news_article_id(token: str) -> str | None:
    try:
        decoded = base64.urlsafe_b64decode(token + ("=" * (-len(token) % 4)))
    except (ValueError, binascii.Error):
        return None

    decoded_text = decoded.decode("utf-8", errors="ignore")
    match = GOOGLE_NEWS_ID_PATTERN.search(decoded_text)
    return match.group(0) if match else None


def extract_bracketed_array(text: str, start_index: int) -> str | None:
    depth = 0
    in_string = False
    escaped = False

    for index in range(start_index, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return text[start_index : index + 1]

    return None


def extract_garturlreq_payload(text: str, token: str) -> str | None:
    if not isinstance(text, str):
        return None

    decoded_text = unescape(text)
    search_start = 0
    while True:
        payload_start = decoded_text.find('["garturlreq"', search_start)
        if payload_start == -1:
            return None

        payload = extract_bracketed_array(decoded_text, payload_start)
        if payload and token in payload:
            return payload
        search_start = payload_start + 1


def build_google_news_batch_payload(inner_payload: str) -> str:
    return json.dumps(
        [[[GOOGLE_NEWS_DECODE_RPC_ID, inner_payload, None, "generic"]]],
        separators=(",", ":"),
    )


def build_google_news_signed_batch_payload(inner_payload: str) -> str:
    return json.dumps(
        [[[GOOGLE_NEWS_DECODE_RPC_ID, inner_payload]]],
        separators=(",", ":"),
    )


def build_signed_garturlreq_payload(
    token: str, *, timestamp: str, signature: str
) -> str | None:
    if not timestamp.isdigit() or not signature:
        return None

    inner_payload = json.dumps(
        [
            "garturlreq",
            [
                [
                    "X",
                    "X",
                    ["X", "X"],
                    None,
                    None,
                    1,
                    1,
                    "US:en",
                    None,
                    1,
                    None,
                    None,
                    None,
                    None,
                    None,
                    0,
                    1,
                ],
                "X",
                "X",
                1,
                [1, 1, 1],
                1,
                1,
                None,
                0,
                0,
                None,
                0,
            ],
            token,
            int(timestamp),
            signature,
        ],
        separators=(",", ":"),
    )
    return inner_payload


def build_fallback_garturlreq_payload(token: str) -> str:
    inner_payload = json.dumps(
        [
            "garturlreq",
            [
                [
                    "en-US",
                    "US",
                    ["FINANCE_TOP_INDICES", "WEB_TEST_1_0_0"],
                    None,
                    None,
                    1,
                    1,
                    "US:en",
                    None,
                    180,
                    None,
                    None,
                    None,
                    None,
                    None,
                    0,
                    None,
                    None,
                    [1608992183, 723341000],
                ],
                "en-US",
                "US",
                1,
                [2, 3, 4, 8],
                1,
                0,
                "655000234",
                0,
                0,
                None,
                0,
            ],
            token,
        ],
        separators=(",", ":"),
    )
    return inner_payload


def extract_google_news_decode_params(text: str) -> tuple[str, str] | None:
    if not isinstance(text, str):
        return None

    signature_match = GOOGLE_NEWS_SIGNATURE_PATTERN.search(text)
    timestamp_match = GOOGLE_NEWS_TIMESTAMP_PATTERN.search(text)
    if not signature_match or not timestamp_match:
        return None

    timestamp = timestamp_match.group(1)
    signature = unescape(signature_match.group(1))
    if not timestamp or not signature:
        return None
    return timestamp, signature


def decode_signed_news_url_with_google(
    token: str, *, timestamp: str, signature: str, timeout_seconds: int
) -> str | None:
    inner_payload = build_signed_garturlreq_payload(
        token,
        timestamp=timestamp,
        signature=signature,
    )
    if not inner_payload:
        return None

    body = "f.req=" + quote(build_google_news_signed_batch_payload(inner_payload))
    try:
        response = requests.post(
            GOOGLE_NEWS_BATCH_EXECUTE_URL,
            data=body,
            timeout=min(timeout_seconds, 10),
            headers={
                "User-Agent": GOOGLE_NEWS_BROWSER_USER_AGENT,
                "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
            },
        )
        response.raise_for_status()
    except requests.RequestException:
        return None

    return extract_first_publisher_url(response.text)


def decode_news_url_with_google(
    inner_payload: str, *, timeout_seconds: int
) -> str | None:
    try:
        response = requests.post(
            GOOGLE_NEWS_BATCH_EXECUTE_URL,
            params={"rpcids": GOOGLE_NEWS_DECODE_RPC_ID},
            data={"f.req": build_google_news_batch_payload(inner_payload)},
            timeout=min(timeout_seconds, 10),
            headers={
                "User-Agent": USER_AGENT,
                "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
            },
        )
        response.raise_for_status()
    except requests.RequestException:
        return None

    return extract_first_publisher_url(response.text)


def resolve_news_url(url: str, *, timeout_seconds: int) -> str:
    cleaned_url = url.strip()
    if not cleaned_url or is_publisher_url(cleaned_url):
        return cleaned_url
    if not is_google_url(cleaned_url):
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
    if is_publisher_url(resolved_url):
        return resolved_url

    token = extract_google_news_token(cleaned_url) or extract_google_news_token(
        resolved_url
    )
    if not token:
        return cleaned_url

    embedded_url = decode_embedded_news_url(token)
    if embedded_url:
        return embedded_url

    decode_params = extract_google_news_decode_params(response.text)
    if decode_params:
        timestamp, signature = decode_params
        signed_decoded_url = decode_signed_news_url_with_google(
            token,
            timestamp=timestamp,
            signature=signature,
            timeout_seconds=timeout_seconds,
        )
        if signed_decoded_url:
            return signed_decoded_url

    signed_payload = extract_garturlreq_payload(response.text, token)
    if signed_payload:
        signed_decoded_url = decode_news_url_with_google(
            signed_payload,
            timeout_seconds=timeout_seconds,
        )
        if signed_decoded_url:
            return signed_decoded_url

    decode_token = decode_google_news_article_id(token) or token
    decoded_url = decode_news_url_with_google(
        build_fallback_garturlreq_payload(decode_token),
        timeout_seconds=timeout_seconds,
    )
    return decoded_url or cleaned_url


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
