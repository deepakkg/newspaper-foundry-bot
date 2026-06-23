from __future__ import annotations

import base64
import tempfile
import unittest
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote
from unittest.mock import MagicMock, patch

import requests
from openai import OpenAIError

import notifications
import tweet_generator
from bluesky_publisher import build_bluesky_post_url, post_to_bluesky
from config import load_config
from discord_sender import send_discord_embed, send_discord_message
from generator import (
    build_compact_prompt,
    build_minimal_prompt,
    build_prompt,
    build_topic_hint,
    normalize_topic,
    request_tweet,
    validate_tweet,
)
from google_news_resolver import resolve_news_url
from link_preview import fetch_link_card_metadata
from logger import (
    PlatformLogResult,
    append_log_entry,
    build_failure_telegram_summary,
    build_telegram_summary,
    build_tweet_log_entry,
)
from news_fetcher import (
    NewsItem,
    build_google_news_rss_url,
    fetch_latest_news,
    parse_rss_items,
    strip_html,
)
from publisher import build_post_text, build_post_text_without_url, max_generated_text_chars
from support import chat_response, load_temp_config, write_env_file
from telegram_sender import send_telegram_message


class LoggerTests(unittest.TestCase):
    def test_build_tweet_log_entry_is_markdown(self) -> None:
        entry = build_tweet_log_entry(
            topic="coffee",
            tone="witty",
            tweet_text="Coffee is back.",
            time_taken_seconds=12.34,
            attempts=2,
            tweet_url="https://x.com/example/status/1",
            timestamp="2026-05-15 12:00:00 IST",
        )

        self.assertIn("## Post published", entry)
        self.assertIn("- Topic: coffee", entry)
        self.assertIn("> Coffee is back.", entry)

    def test_append_log_entry_writes_markdown_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "tweet-history.md"
            entry = build_tweet_log_entry(
                topic="coffee",
                tone="witty",
                tweet_text="Coffee is back.",
                time_taken_seconds=12.34,
                attempts=2,
                tweet_url="https://x.com/example/status/1",
                timestamp="2026-05-15 12:00:00 IST",
            )

            append_log_entry(log_path, entry)
            content = log_path.read_text(encoding="utf-8")

        self.assertIn("# Post History", content)
        self.assertIn("Topic: coffee", content)

    def test_build_tweet_log_entry_includes_news_metadata(self) -> None:
        entry = build_tweet_log_entry(
            topic="ai agents",
            tone="serious",
            tweet_text="AI agents are moving from demos into support queues.",
            time_taken_seconds=3.21,
            attempts=1,
            tweet_url="https://x.com/example/status/2",
            news_title="AI agents reshape support workflows",
            news_source="Example News",
            news_published_at="2026-05-31 15:30 IST",
            news_url="https://example.com/ai-agents",
        )

        self.assertIn("News title: AI agents reshape support workflows", entry)
        self.assertIn("News source: Example News", entry)
        self.assertIn("News published: 2026-05-31 15:30 IST", entry)
        self.assertIn("News URL: https://example.com/ai-agents", entry)

    def test_build_telegram_summary_excludes_full_log_and_url(self) -> None:
        summary = build_telegram_summary(
            topic="coffee",
            tone="witty",
            tweet_text="Coffee is back.",
            time_taken_seconds=12.34,
            attempts=2,
        )

        self.assertIn("Topic: coffee", summary)
        self.assertIn("Post text:\nCoffee is back.", summary)
        self.assertNotIn("# Post History", summary)
        self.assertNotIn("Post URL:", summary)

    def test_build_telegram_summary_includes_news_reference_when_provided(self) -> None:
        summary = build_telegram_summary(
            topic="ai agents",
            tone="serious",
            tweet_text="AI agents are moving. 🤖 #botWrites https://example.com/news",
            time_taken_seconds=3.21,
            attempts=1,
            news_title="AI agents reshape support workflows",
            news_source="Example News",
            news_url="https://example.com/news",
            news_published_at="2026-05-31 15:30 IST",
        )

        self.assertIn("News reference:", summary)
        self.assertIn("AI agents reshape support workflows (Example News)", summary)
        self.assertIn("Published: 2026-05-31 15:30 IST", summary)
        self.assertNotIn("\nhttps://example.com/news\n", summary)
        self.assertIn(
            "AI agents are moving. 🤖 #botWrites https://example.com/news", summary
        )

    def test_build_failure_telegram_summary_includes_error_and_news_reference(
        self,
    ) -> None:
        summary = build_failure_telegram_summary(
            topic="ai agents",
            tone="serious",
            error_message="Generation failed",
            news_title="AI agents reshape support workflows",
            news_source="Example News",
            news_url="https://example.com/news",
            news_published_at="2026-05-31 15:30 IST",
        )

        self.assertIn("Content bot failed", summary)
        self.assertIn("Generation failed", summary)
        self.assertIn("AI agents reshape support workflows (Example News)", summary)
        self.assertIn("Published: 2026-05-31 15:30 IST", summary)
        self.assertIn("https://example.com/news", summary)

    def test_format_news_published_at_converts_utc_to_ist(self) -> None:
        news_item = NewsItem(
            title="Journeo Secures Metroline Manchester Deal",
            source="Yahoo Finance UK",
            published_at=datetime(2026, 6, 23, 8, 24, tzinfo=timezone.utc),
            link="https://example.com/news",
            summary="A transport technology deal was announced.",
        )

        self.assertEqual(
            notifications.format_news_published_at(news_item),
            "2026-06-23 13:54 IST",
        )

    def test_notification_and_log_copy_uses_generic_post_language(self) -> None:
        news_item = NewsItem(
            title="AI agents reshape support workflows",
            source="Example News",
            published_at=datetime(2026, 5, 31, 10, 0, tzinfo=timezone.utc),
            link="https://example.com/news",
            summary="Companies are deploying agents to resolve support tickets.",
        )
        rendered_outputs = [
            build_tweet_log_entry(
                topic="ai agents",
                tone="serious",
                tweet_text="AI agents are moving. 🤖 #botWrites",
                time_taken_seconds=3.21,
                attempts=1,
                tweet_url="https://example.com/post",
            ),
            build_telegram_summary(
                topic="ai agents",
                tone="serious",
                tweet_text="AI agents are moving. 🤖 #botWrites",
                time_taken_seconds=3.21,
                attempts=1,
            ),
            build_failure_telegram_summary(
                topic="ai agents",
                tone="serious",
                error_message="Generation failed",
            ),
            str(
                notifications.build_discord_success_embed(
                    topic="ai agents",
                    tone="serious",
                    tweet_text="AI agents are moving. 🤖 #botWrites",
                    time_taken_seconds=3.21,
                    attempts=1,
                    news_item=news_item,
                )
            ),
            str(
                notifications.build_discord_manual_embed(
                    topic="ai agents",
                    tone="serious",
                    time_taken_seconds=3.21,
                    attempts=1,
                    news_item=news_item,
                )
            ),
            str(
                notifications.build_discord_failure_embed(
                    topic="ai agents",
                    tone="serious",
                    news_item=news_item,
                    error_message="Generation failed",
                )
            ),
        ]

        for output in rendered_outputs:
            self.assertNotIn("Tweet", output)
            self.assertNotIn("tweet", output)
        self.assertIn("Post published", rendered_outputs[3])
        self.assertIn("Final post", rendered_outputs[3])
        self.assertIn("Post ready", rendered_outputs[4])
        self.assertIn("Content bot failed", rendered_outputs[5])

    def test_discord_success_embed_orders_platform_results_before_final_post(self) -> None:
        news_item = NewsItem(
            title="AI agents reshape support workflows",
            source="Example News",
            published_at=datetime(2026, 5, 31, 10, 0, tzinfo=timezone.utc),
            link="https://example.com/ai-agents",
            summary="Companies are deploying agents to resolve support tickets.",
        )

        embed = notifications.build_discord_success_embed(
            topic="ai agents",
            tone="analysis",
            tweet_text="AI agents need better handoffs. 🤖 #botWrites",
            time_taken_seconds=3.21,
            attempts=1,
            news_item=news_item,
            platform_results=[
                PlatformLogResult(
                    platform="Bluesky",
                    status="published",
                    url="https://bsky.app/profile/example/post/abc",
                    identifier="at://did:plc:abc/app.bsky.feed.post/abc",
                ),
                PlatformLogResult(
                    platform="Instagram",
                    status="published",
                    identifier="18050706419551949",
                ),
            ],
        )

        field_names = [field["name"] for field in embed["fields"]]
        self.assertEqual(
            field_names,
            [
                "Topic",
                "Tone",
                "Attempts",
                "Time taken",
                "News title",
                "News source",
                "News published",
                "Platform results",
                "Final post",
            ],
        )
        self.assertEqual(
            embed["fields"][4]["value"],
            "AI agents reshape support workflows",
        )
        self.assertEqual(embed["fields"][5]["value"], "Example News")
        self.assertEqual(embed["fields"][6]["value"], "2026-05-31 15:30 IST")
        platform_results = embed["fields"][7]["value"]
        self.assertIn(
            "Bluesky: published | https://bsky.app/profile/example/post/abc | "
            "at://did:plc:abc/app.bsky.feed.post/abc",
            platform_results,
        )
        self.assertIn(
            "Instagram: published | 18050706419551949",
            platform_results,
        )

    def test_discord_manual_and_failure_embeds_use_ist_news_time(self) -> None:
        news_item = NewsItem(
            title="AI agents reshape support workflows",
            source="Example News",
            published_at=datetime(2026, 5, 31, 10, 0, tzinfo=timezone.utc),
            link="https://example.com/ai-agents",
            summary="Companies are deploying agents to resolve support tickets.",
        )

        manual_embed = notifications.build_discord_manual_embed(
            topic="ai agents",
            tone="analysis",
            time_taken_seconds=3.21,
            attempts=1,
            news_item=news_item,
        )
        failure_embed = notifications.build_discord_failure_embed(
            topic="ai agents",
            tone="analysis",
            news_item=news_item,
            error_message="Publishing failed",
        )

        manual_values = {
            field["name"]: field["value"] for field in manual_embed["fields"]
        }
        failure_values = {
            field["name"]: field["value"] for field in failure_embed["fields"]
        }
        self.assertEqual(manual_values["News published"], "2026-05-31 15:30 IST")
        self.assertEqual(failure_values["News published"], "2026-05-31 15:30 IST")
