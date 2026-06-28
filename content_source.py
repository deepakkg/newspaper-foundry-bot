from __future__ import annotations

import random
from dataclasses import dataclass

from config import AppConfig
from news_fetcher import NewsItem, fetch_latest_news
from on_demand_requests import OnDemandRequest, fetch_next_on_demand_request


@dataclass(frozen=True)
class ContentSource:
    topic: str
    tone: str
    news_item: NewsItem | None
    on_demand_request: OnDemandRequest | None = None


def select_content_source(config: AppConfig) -> ContentSource:
    on_demand_request: OnDemandRequest | None = None
    if config.on_demand_requests_enabled:
        try:
            on_demand_request = fetch_next_on_demand_request(config)
        except Exception as exc:
            print(f"Warning: Discord on-demand request lookup failed: {exc}")

    if on_demand_request is not None:
        topic = on_demand_request.topic
        tone = on_demand_request.tone or random.choice(config.tones)
        news_item = on_demand_request.news_item
        if on_demand_request.kind == "direct_post":
            print("Using on-demand direct post request.")
        elif news_item is not None:
            print(f"Using on-demand news URL: {news_item.title} ({news_item.source})")
        return ContentSource(
            topic=topic,
            tone=tone,
            news_item=news_item,
            on_demand_request=on_demand_request,
        )

    topic = random.choice(config.topics)
    tone = random.choice(config.tones)
    news_item = None

    if config.news_enabled:
        try:
            news_item = fetch_latest_news(topic, config)
            if news_item is None:
                print(f"No recent RSS news found for {topic}. Using generic topic prompt.")
            else:
                print(f"Using RSS news: {news_item.title} ({news_item.source})")
        except Exception as exc:
            print(f"Warning: RSS news lookup failed for {topic}: {exc}")

    return ContentSource(
        topic=topic,
        tone=tone,
        news_item=news_item,
        on_demand_request=None,
    )
