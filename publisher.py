from __future__ import annotations

from dataclasses import dataclass

import requests
from requests_oauthlib import OAuth1

from config import AppConfig


X_POST_URL = "https://api.x.com/2/tweets"
POST_HASHTAG = "#botWrites"


@dataclass(frozen=True)
class PublishedTweet:
    tweet_id: str
    url: str


def build_post_text(tweet_text: str) -> str:
    cleaned = tweet_text.rstrip()
    if cleaned.endswith(POST_HASHTAG):
        return cleaned
    return f"{cleaned} {POST_HASHTAG}"


def post_tweet_to_x(config: AppConfig, tweet_text: str) -> PublishedTweet:
    post_text = build_post_text(tweet_text)
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
