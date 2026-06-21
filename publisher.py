from __future__ import annotations

from dataclasses import dataclass

import requests
from requests_oauthlib import OAuth1

from config import AppConfig


X_POST_URL = "https://api.x.com/2/tweets"
POST_HASHTAG = "#botWrites"
X_MAX_POST_CHARS = 280
X_URL_CHAR_COUNT = 23


@dataclass(frozen=True)
class PublishedTweet:
    tweet_id: str
    url: str


def _clean_post_part(text: str) -> str:
    return " ".join(text.strip().split())


def build_post_text(tweet_text: str, news_url: str | None = None) -> str:
    cleaned_url = _clean_post_part(news_url or "")
    parts = [
        part
        for part in _clean_post_part(tweet_text).split()
        if part != POST_HASHTAG and part != cleaned_url
    ]
    suffix = [POST_HASHTAG]
    if cleaned_url:
        suffix.append(cleaned_url)
    return " ".join([*parts, *suffix])


def build_post_text_without_url(tweet_text: str) -> str:
    return build_post_text(tweet_text, news_url=None)


def reserved_post_chars(news_url: str | None = None) -> int:
    reserved = len(f" {POST_HASHTAG}")
    if _clean_post_part(news_url or ""):
        reserved += 1 + X_URL_CHAR_COUNT
    return reserved


def max_generated_text_chars(
    configured_max_chars: int, news_url: str | None = None
) -> int:
    available = X_MAX_POST_CHARS - reserved_post_chars(news_url)
    return max(1, min(configured_max_chars, available))


def post_tweet_to_x(
    config: AppConfig, tweet_text: str, news_url: str | None = None
) -> PublishedTweet:
    post_text = build_post_text(tweet_text, news_url)
    auth = OAuth1(
        config.x_api_key,
        config.x_api_key_secret,
        config.x_access_token,
        config.x_access_token_secret,
    )
    response = requests.post(
        X_POST_URL,
        auth=auth,
        json={"text": post_text},
        timeout=config.timeout_seconds,
    )
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError("X API returned a non-JSON response.") from exc

    if response.status_code >= 400:
        detail = payload.get("detail") or payload.get("title") or str(payload)
        raise RuntimeError(f"X posting failed: {detail}")

    tweet_id = payload.get("data", {}).get("id")
    if not tweet_id:
        raise RuntimeError("X API response did not include a tweet ID.")

    tweet_url = f"https://x.com/{config.x_username}/status/{tweet_id}"
    return PublishedTweet(tweet_id=tweet_id, url=tweet_url)
