from __future__ import annotations

from instagram_hashtags import BOT_HASHTAG, hashtag_from_text, normalize_hashtag
from news_fetcher import NewsItem
from time_formatting import format_datetime_ist


SOURCE_SUFFIX_SEPARATORS = (" - ", " – ", " — ")


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
