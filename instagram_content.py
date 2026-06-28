from __future__ import annotations

import re

from openai import OpenAI

from config import AppConfig
from generator import extract_response_text, request_completion
from news_fetcher import NewsItem
from time_formatting import format_datetime_ist

BOT_HASHTAG = "#botWrites"
HASHTAG_PATTERN = re.compile(r"#[A-Za-z][A-Za-z0-9_]{1,40}")
SOURCE_SUFFIX_SEPARATORS = (" - ", " – ", " — ")


def normalize_hashtag(value: str) -> str | None:
    cleaned = re.sub(r"[^A-Za-z0-9_ ]+", "", value.strip())
    compact = "".join(cleaned.split())
    if not compact:
        return None
    if compact.startswith("#"):
        compact = compact[1:]
    if not compact or not compact[0].isalpha():
        return None
    return f"#{compact[:40]}"


def hashtag_from_text(value: str) -> str | None:
    return normalize_hashtag(value)


def extract_hashtags(text: str) -> list[str]:
    found = HASHTAG_PATTERN.findall(text)
    normalized: list[str] = []
    seen: set[str] = set()
    for hashtag in found:
        cleaned = normalize_hashtag(hashtag)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key == BOT_HASHTAG.lower() or key in seen:
            continue
        seen.add(key)
        normalized.append(cleaned)
    return normalized[:6]


def fallback_hashtags(topic: str, tone: str) -> list[str]:
    candidates = [
        hashtag_from_text(topic),
        hashtag_from_text(tone),
        "#News",
        "#Analysis",
    ]
    result: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate:
            continue
        key = candidate.lower()
        if key == BOT_HASHTAG.lower() or key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result


def fallback_hashtags_from_text(post_text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9]{2,}", post_text)
    result: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        cleaned = normalize_hashtag(token)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key == BOT_HASHTAG.lower() or key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
        if len(result) >= 4:
            break
    return result


def build_hashtag_prompt(topic: str, tone: str, news_item: NewsItem | None) -> str:
    news_block = ""
    if news_item:
        news_block = f"""
News context:
- Title: {news_item.title}
- Source: {news_item.source}
- Summary: {news_item.summary or news_item.title}
"""
    return f"""Suggest 3 to 6 Instagram hashtags for this post.
Topic: {topic}
Tone: {tone}
{news_block}
Rules:
- Return only hashtags separated by spaces.
- Use simple discoverable Instagram hashtags.
- Do not include #botWrites.
- Do not include URLs, explanations, labels, or quotes.
"""


def build_text_hashtag_prompt(post_text: str) -> str:
    return f"""Suggest 3 to 6 Instagram hashtags for this post text.
Post text:
{post_text}

Rules:
- Return only hashtags separated by spaces.
- Use simple discoverable Instagram hashtags.
- Do not include #botWrites.
- Do not include URLs, explanations, labels, or quotes.
"""


def generate_instagram_hashtags(
    client: OpenAI,
    config: AppConfig,
    topic: str,
    tone: str,
    news_item: NewsItem | None,
) -> list[str]:
    try:
        response = request_completion(
            config=config,
            client=client,
            prompt=build_hashtag_prompt(topic, tone, news_item),
        )
        hashtags = extract_hashtags(extract_response_text(response))
    except Exception:
        hashtags = []
    return hashtags or fallback_hashtags(topic, tone)


def generate_instagram_hashtags_from_text(
    client: OpenAI,
    config: AppConfig,
    post_text: str,
) -> list[str]:
    try:
        response = request_completion(
            config=config,
            client=client,
            prompt=build_text_hashtag_prompt(post_text),
        )
        hashtags = extract_hashtags(extract_response_text(response))
    except Exception:
        hashtags = []
    return hashtags or fallback_hashtags_from_text(post_text)


def format_news_published(news_item: NewsItem | None) -> str | None:
    if not news_item:
        return None
    return format_datetime_ist(news_item.published_at)


def format_caption_news_title(title: str, source: str) -> str:
    cleaned_title = " ".join(title.split())
    cleaned_source = " ".join(source.split())
    if not cleaned_title:
        return "Not available"
    if not cleaned_source:
        return cleaned_title

    for separator in SOURCE_SUFFIX_SEPARATORS:
        suffix = f"{separator}{cleaned_source}"
        if cleaned_title.lower().endswith(suffix.lower()):
            return cleaned_title[: -len(suffix)].rstrip()
    return cleaned_title


def build_instagram_caption(
    *,
    topic: str,
    tone: str,
    news_item: NewsItem | None,
    llm_hashtags: list[str],
    article_link_in_bio: bool = False,
    include_topic_tone_hashtags: bool = True,
) -> str:
    published = format_news_published(news_item)
    if news_item:
        lines: list[str] = [
            format_caption_news_title(news_item.title, news_item.source),
            f"Source: {news_item.source or 'Not available'}",
            f"Published At: {published or 'Not available'}",
        ]
        if article_link_in_bio:
            lines.append("Article link in bio.")
    else:
        lines = []

    hashtags: list[str] = []
    seen: set[str] = set()
    candidates = [*llm_hashtags, BOT_HASHTAG]
    if include_topic_tone_hashtags:
        candidates = [
            hashtag_from_text(topic),
            hashtag_from_text(tone),
            *candidates,
        ]
    for candidate in candidates:
        cleaned = normalize_hashtag(candidate or "")
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        hashtags.append(cleaned)

    hashtags = [tag for tag in hashtags if tag.lower() != BOT_HASHTAG.lower()]
    hashtags = hashtags[:11]
    hashtags.append(BOT_HASHTAG)
    if lines:
        lines.append("")
    lines.append(" ".join(hashtags))
    return "\n".join(lines)
