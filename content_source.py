from __future__ import annotations

import random
from dataclasses import dataclass, field

from config import AppConfig
from news_fetcher import NewsItem, fetch_latest_news
from on_demand_requests import OnDemandRequest, fetch_next_on_demand_request
from run_state import RunStateStore


@dataclass(frozen=True)
class ContentSource:
    topic: str
    tone: str
    news_item: NewsItem | None
    on_demand_request: OnDemandRequest | None = None
    recent_post_texts: list[str] = field(default_factory=list)


def choose_novel_topic(topics: list[str], recent_runs: list[dict[str, object]]) -> str:
    counts = {topic: 0 for topic in topics}
    for run in recent_runs:
        recent_topic = str(run.get("topic", ""))
        if recent_topic in counts:
            counts[recent_topic] += 1
    minimum = min(counts.values())
    candidates = [topic for topic, count in counts.items() if count == minimum]
    return random.choice(candidates)


def choose_novel_tone(tones: list[str], recent_runs: list[dict[str, object]]) -> str:
    counts = {tone: 0 for tone in tones}
    for run in recent_runs:
        recent_tone = str(run.get("tone", ""))
        if recent_tone in counts:
            counts[recent_tone] += 1
    minimum = min(counts.values())
    candidates = [tone for tone, count in counts.items() if count == minimum]
    return random.choice(candidates)


def select_content_source(config: AppConfig) -> ContentSource:
    state_store = RunStateStore(config.state_file_path)
    recent_runs = state_store.recent_runs()
    recent_post_texts = state_store.recent_post_texts()
    on_demand_request: OnDemandRequest | None = None
    if config.on_demand_requests_enabled:
        try:
            on_demand_request = fetch_next_on_demand_request(config)
        except Exception as exc:
            print(f"Warning: Discord on-demand request lookup failed: {exc}")

    if on_demand_request is not None:
        topic = on_demand_request.topic
        tone = on_demand_request.tone or choose_novel_tone(config.tones, recent_runs)
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
            recent_post_texts=recent_post_texts,
        )

    topic = choose_novel_topic(config.topics, recent_runs)
    tone = choose_novel_tone(config.tones, recent_runs)
    news_item = None

    if config.news_enabled:
        try:
            excluded_urls = state_store.recent_news_urls()
            news_item = (
                fetch_latest_news(topic, config, excluded_urls=excluded_urls)
                if excluded_urls
                else fetch_latest_news(topic, config)
            )
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
        recent_post_texts=recent_post_texts,
    )
