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


class GeneratorValidationTests(unittest.TestCase):
    def test_build_prompt_stays_compact(self) -> None:
        prompt = build_prompt("saas professional services", "serious", 230, 1)

        self.assertIn("Write one post about:", prompt)
        self.assertLess(len(prompt), 1200)

    def test_build_compact_prompt_is_shorter(self) -> None:
        full_prompt = build_prompt("saas professional services", "serious", 230, 2)
        compact_prompt = build_compact_prompt(
            "saas professional services", "serious", 230, 2
        )

        self.assertLess(len(compact_prompt), len(full_prompt))
        self.assertLess(len(compact_prompt), 400)
        self.assertIn("Use 1 or 2 relevant emojis.", compact_prompt)
        self.assertIn("no article URL", compact_prompt)

    def test_build_minimal_prompt_is_shorter_than_compact(self) -> None:
        compact_prompt = build_compact_prompt(
            "saas professional services", "serious", 230, 2
        )
        minimal_prompt = build_minimal_prompt(
            "saas professional services", "serious", 230
        )

        self.assertLess(len(minimal_prompt), len(compact_prompt))
        self.assertLess(len(minimal_prompt), 120)
        self.assertIn("Add 1-2 emojis.", minimal_prompt)
        self.assertIn("No hashtag/link.", minimal_prompt)

    def test_build_topic_hint_shortens_long_topic(self) -> None:
        self.assertEqual(
            build_topic_hint("saas professional services"),
            "saas professional",
        )

    def test_build_prompt_includes_news_context(self) -> None:
        news_item = NewsItem(
            title="AI agents reshape enterprise workflows",
            source="Example News",
            published_at=datetime(2026, 5, 31, 12, 0, tzinfo=timezone.utc),
            link="https://example.com/ai-agents",
            summary="Companies are using AI agents to automate multi-step support work.",
        )

        prompt = build_prompt("ai agents", "serious", 230, 1, news_item)

        self.assertIn("Current news context:", prompt)
        self.assertIn("AI agents reshape enterprise workflows", prompt)
        self.assertIn("Example News", prompt)
        self.assertIn("Include 1 or 2 relevant emojis.", prompt)
        self.assertIn("Do not include the article URL.", prompt)

    def test_build_prompt_includes_deepak_style_guidance(self) -> None:
        prompt = build_prompt("leadership", "serious", 230, 1)

        self.assertIn("Write like Deepak", prompt)
        self.assertIn("direct, practical, concise", prompt)
        self.assertIn("Do not force first person", prompt)
        self.assertIn("Pseudo-profound", prompt)
        self.assertIn("The real lesson", prompt)
        self.assertIn("More than two emojis", prompt)

    def test_request_tweet_retries_with_compact_prompt_on_context_error(self) -> None:
        tmp_dir, config = load_temp_config()
        self.addCleanup(tmp_dir.cleanup)
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            RuntimeError("prompt too long; exceeded max context length by 8 tokens"),
            chat_response(
                "SaaS professional services still win when handoff work is treated like product, not overhead."
            ),
        ]

        tweet = request_tweet(
            client,
            config,
            "saas professional services",
            "serious",
            1,
        )

        self.assertIn("SaaS professional services", tweet)
        self.assertEqual(client.chat.completions.create.call_count, 2)
        first_prompt = client.chat.completions.create.call_args_list[0].kwargs[
            "messages"
        ][0]["content"]
        second_prompt = client.chat.completions.create.call_args_list[1].kwargs[
            "messages"
        ][0]["content"]
        self.assertEqual(
            client.chat.completions.create.call_args_list[0].kwargs["model"],
            config.llm_model,
        )
        self.assertLess(len(second_prompt), len(first_prompt))

    def test_request_tweet_retries_with_minimal_prompt_if_needed(self) -> None:
        tmp_dir, config = load_temp_config()
        self.addCleanup(tmp_dir.cleanup)
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            RuntimeError("prompt too long; exceeded max context length by 8 tokens"),
            RuntimeError("prompt too long; exceeded max context length by 3 tokens"),
            chat_response(
                "SaaS services become more valuable when messy implementation work is handled well."
            ),
        ]

        tweet = request_tweet(
            client,
            config,
            "saas professional services",
            "serious",
            1,
        )

        self.assertIn("SaaS services", tweet)
        self.assertEqual(client.chat.completions.create.call_count, 3)
        prompts = [
            call.kwargs["messages"][0]["content"]
            for call in client.chat.completions.create.call_args_list
        ]
        self.assertLess(len(prompts[1]), len(prompts[0]))
        self.assertLess(len(prompts[2]), len(prompts[1]))
        self.assertIn("Post about saas professional.", prompts[2])

    def test_request_tweet_rejects_empty_chat_response(self) -> None:
        tmp_dir, config = load_temp_config()
        self.addCleanup(tmp_dir.cleanup)
        client = MagicMock()
        client.chat.completions.create.return_value = chat_response("")

        with self.assertRaisesRegex(
            RuntimeError, "Server response did not include a valid post"
        ):
            request_tweet(client, config, "coffee", "witty", 1)

    def test_accepts_specific_topic_relevant_tweet(self) -> None:
        topic, topic_tokens = normalize_topic("Narendra Modi")
        tweet = "Narendra Modi keeps turning routine policy announcements into headline events, and that timing is half the story. 🗳️"

        result = validate_tweet(
            tweet,
            topic,
            topic_tokens,
            max_tweet_chars=230,
            attempt_number=1,
            max_retries=5,
        )

        self.assertIsNone(result)

    def test_accepts_two_relevant_emojis(self) -> None:
        topic, topic_tokens = normalize_topic("ai agents")
        tweet = "AI agents are moving from demos into support queues, where handoffs and escalation paths now matter. 🤖⚙️"

        result = validate_tweet(
            tweet,
            topic,
            topic_tokens,
            max_tweet_chars=230,
            attempt_number=1,
            max_retries=5,
        )

        self.assertIsNone(result)

    def test_rejects_tweet_without_emoji(self) -> None:
        topic, topic_tokens = normalize_topic("ai agents")
        tweet = "AI agents are moving from demos into support queues, where handoffs and escalation paths now matter."

        result = validate_tweet(
            tweet,
            topic,
            topic_tokens,
            max_tweet_chars=230,
            attempt_number=1,
            max_retries=5,
        )

        self.assertEqual(result, "missing emoji")

    def test_rejects_more_than_two_emojis(self) -> None:
        topic, topic_tokens = normalize_topic("ai agents")
        tweet = "AI agents are moving from demos into support queues, where handoffs and escalation paths now matter. 🤖⚙️🚀"

        result = validate_tweet(
            tweet,
            topic,
            topic_tokens,
            max_tweet_chars=230,
            attempt_number=1,
            max_retries=5,
        )

        self.assertEqual(result, "too many emojis")

    def test_rejects_generic_coffee_tweet(self) -> None:
        topic, topic_tokens = normalize_topic("coffee")
        tweet = "My morning coffee is definitely hitting the spot. Makes it easier to face the day. ☕"

        result = validate_tweet(
            tweet,
            topic,
            topic_tokens,
            max_tweet_chars=230,
            attempt_number=1,
            max_retries=5,
        )

        self.assertEqual(result, "too generic")

    def test_rejects_pseudo_profound_tweet(self) -> None:
        topic, topic_tokens = normalize_topic("ai agents")
        tweet = "AI agents are not about automation, it's about unlocking the real lesson of human potential. 🤖"

        result = validate_tweet(
            tweet,
            topic,
            topic_tokens,
            max_tweet_chars=230,
            attempt_number=1,
            max_retries=5,
        )

        self.assertEqual(result, "pseudo-profound phrasing")
