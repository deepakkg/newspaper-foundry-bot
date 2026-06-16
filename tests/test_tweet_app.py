from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

from config import load_config
from generator import (
    build_compact_prompt,
    build_minimal_prompt,
    build_prompt,
    build_topic_hint,
    normalize_topic,
    request_tweet,
    validate_tweet,
)
from logger import (
    append_tweet_log,
    build_failure_telegram_summary,
    build_slot_marker,
    build_telegram_summary,
    build_tweet_log_entry,
    has_logged_slot,
)
from ollama import ResponseError
from news_fetcher import (
    NewsItem,
    build_google_news_rss_url,
    fetch_latest_news,
    parse_rss_items,
    resolve_news_url,
    strip_html,
)
from publisher import build_post_text, max_generated_text_chars
from telegram_sender import send_telegram_message
import tweet_generator


def write_env_file(path: Path, **overrides: str) -> None:
    values = {
        "OLLAMA_HOST": "http://localhost:11434",
        "TOPICS": "coffee,learning",
        "TONES": "witty,serious",
        "POST_TO_X": "false",
        "NEWS_ENABLED": "false",
        "NEWS_RECENCY_HOURS": "48",
        "NEWS_REGION": "US",
        "NEWS_LANGUAGE": "en",
        "OLLAMA_API_KEY": "",
        "X_API_KEY": "",
        "X_API_KEY_SECRET": "",
        "X_ACCESS_TOKEN": "",
        "X_ACCESS_TOKEN_SECRET": "",
        "X_USERNAME": "",
        "TELEGRAM_BOT_TOKEN": "",
        "TELEGRAM_CHAT_ID": "",
    }
    values.update(overrides)
    path.write_text(
        "\n".join(f"{key}={value}" for key, value in values.items()) + "\n",
        encoding="utf-8",
    )


def load_temp_config(**overrides: str):
    tmp_dir = tempfile.TemporaryDirectory()
    env_path = Path(tmp_dir.name) / ".env"
    overrides.setdefault("LOG_FILE_PATH", str(Path(tmp_dir.name) / "tweet-history.md"))
    write_env_file(env_path, **overrides)
    config = load_config(env_path)
    return tmp_dir, config


class GeneratorValidationTests(unittest.TestCase):
    def test_build_prompt_stays_compact(self) -> None:
        prompt = build_prompt("saas professional services", "serious", 230, 1)

        self.assertIn("Write one tweet about:", prompt)
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
        client.generate.side_effect = [
            ResponseError(
                "prompt too long; exceeded max context length by 8 tokens",
                status_code=400,
            ),
            {"response": "SaaS professional services still win when handoff work is treated like product, not overhead."},
        ]

        tweet = request_tweet(
            client,
            config,
            "saas professional services",
            "serious",
            1,
        )

        self.assertIn("SaaS professional services", tweet)
        self.assertEqual(client.generate.call_count, 2)
        first_prompt = client.generate.call_args_list[0].kwargs["prompt"]
        second_prompt = client.generate.call_args_list[1].kwargs["prompt"]
        self.assertLess(len(second_prompt), len(first_prompt))

    def test_request_tweet_retries_with_minimal_prompt_if_needed(self) -> None:
        tmp_dir, config = load_temp_config()
        self.addCleanup(tmp_dir.cleanup)
        client = MagicMock()
        client.generate.side_effect = [
            ResponseError(
                "prompt too long; exceeded max context length by 8 tokens",
                status_code=400,
            ),
            ResponseError(
                "prompt too long; exceeded max context length by 3 tokens",
                status_code=400,
            ),
            {"response": "SaaS services become more valuable when messy implementation work is handled well."},
        ]

        tweet = request_tweet(
            client,
            config,
            "saas professional services",
            "serious",
            1,
        )

        self.assertIn("SaaS services", tweet)
        self.assertEqual(client.generate.call_count, 3)
        prompts = [call.kwargs["prompt"] for call in client.generate.call_args_list]
        self.assertLess(len(prompts[1]), len(prompts[0]))
        self.assertLess(len(prompts[2]), len(prompts[1]))
        self.assertIn("Tweet about saas professional.", prompts[2])

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


class ConfigTests(unittest.TestCase):
    def test_load_config_accepts_github_style_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "logs" / "tweet-history.md"
            env_path = Path(tmp_dir) / ".env"
            write_env_file(
                env_path,
                POST_TO_X="true",
                X_API_KEY="key",
                X_API_KEY_SECRET="secret",
                X_ACCESS_TOKEN="token",
                X_ACCESS_TOKEN_SECRET="token-secret",
                X_USERNAME="example",
                TELEGRAM_BOT_TOKEN="bot-token",
                TELEGRAM_CHAT_ID="12345",
                LOG_FILE_PATH=str(log_path),
            )

            config = load_config(env_path)

        self.assertTrue(config.post_to_x)
        self.assertFalse(config.news_enabled)
        self.assertEqual(config.news_recency_hours, 48)
        self.assertEqual(config.news_region, "US")
        self.assertEqual(config.news_language, "en")
        self.assertEqual(config.log_file_path, log_path)

    def test_load_config_accepts_news_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            write_env_file(
                env_path,
                NEWS_ENABLED="true",
                NEWS_RECENCY_HOURS="24",
                NEWS_REGION="US",
                NEWS_LANGUAGE="en",
            )

            config = load_config(env_path)

        self.assertTrue(config.news_enabled)
        self.assertEqual(config.news_recency_hours, 24)
        self.assertEqual(config.news_region, "US")
        self.assertEqual(config.news_language, "en")

    def test_load_config_rejects_partial_telegram_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            write_env_file(env_path, TELEGRAM_BOT_TOKEN="bot-token")

            with self.assertRaisesRegex(
                ValueError,
                "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must both be set",
            ):
                load_config(env_path)


class NewsFetcherTests(unittest.TestCase):
    def test_build_google_news_rss_url_uses_us_english_settings(self) -> None:
        url = build_google_news_rss_url("ai agents", language="en", region="US")

        self.assertIn("https://news.google.com/rss/search?", url)
        self.assertIn("q=ai+agents", url)
        self.assertIn("hl=en-US", url)
        self.assertIn("gl=US", url)
        self.assertIn("ceid=US%3Aen", url)

    def test_strip_html_cleans_rss_description(self) -> None:
        cleaned = strip_html("<a>Headline</a>&nbsp;&nbsp;<font>Source</font>")

        self.assertEqual(cleaned, "Headline Source")

    def test_parse_rss_items_filters_stale_items(self) -> None:
        rss = """<?xml version="1.0" encoding="UTF-8" ?>
<rss version="2.0">
  <channel>
    <item>
      <title>Fresh AI agents headline</title>
      <link>https://example.com/fresh</link>
      <description><![CDATA[Fresh <b>summary</b> text.]]></description>
      <pubDate>Sun, 31 May 2026 10:00:00 GMT</pubDate>
      <source url="https://example.com">Example News</source>
    </item>
    <item>
      <title>Old AI agents headline</title>
      <link>https://example.com/old</link>
      <description>Old summary text.</description>
      <pubDate>Thu, 28 May 2026 10:00:00 GMT</pubDate>
      <source url="https://example.com">Example News</source>
    </item>
  </channel>
</rss>
"""
        now = datetime(2026, 5, 31, 12, 0, tzinfo=timezone.utc)

        items = parse_rss_items(rss, now=now, recency_hours=48)

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].title, "Fresh AI agents headline")
        self.assertEqual(items[0].source, "Example News")
        self.assertEqual(items[0].summary, "Fresh summary text.")

    def test_resolve_news_url_returns_publisher_redirect(self) -> None:
        google_url = "https://news.google.com/rss/articles/example"
        response = MagicMock(url="https://example.com/story")

        with patch("news_fetcher.requests.get", return_value=response) as mock_get:
            resolved_url = resolve_news_url(google_url, timeout_seconds=120)

        self.assertEqual(resolved_url, "https://example.com/story")
        mock_get.assert_called_once_with(
            google_url,
            timeout=10,
            headers={"User-Agent": "gemma-tweet-bot/1.0"},
            allow_redirects=True,
        )
        response.raise_for_status.assert_called_once()

    def test_resolve_news_url_keeps_google_url_when_redirect_stays_google(self) -> None:
        google_url = "https://news.google.com/rss/articles/example"
        response = MagicMock(url="https://news.google.com/read/example")

        with patch("news_fetcher.requests.get", return_value=response):
            resolved_url = resolve_news_url(google_url, timeout_seconds=120)

        self.assertEqual(resolved_url, google_url)

    def test_resolve_news_url_keeps_google_url_when_request_fails(self) -> None:
        google_url = "https://news.google.com/rss/articles/example"

        with patch(
            "news_fetcher.requests.get",
            side_effect=requests.RequestException("network failed"),
        ):
            resolved_url = resolve_news_url(google_url, timeout_seconds=120)

        self.assertEqual(resolved_url, google_url)

    def test_resolve_news_url_skips_non_google_url(self) -> None:
        article_url = "https://example.com/story"

        with patch("news_fetcher.requests.get") as mock_get:
            resolved_url = resolve_news_url(article_url, timeout_seconds=120)

        self.assertEqual(resolved_url, article_url)
        mock_get.assert_not_called()

    def test_fetch_latest_news_returns_latest_usable_item(self) -> None:
        tmp_dir, config = load_temp_config(NEWS_ENABLED="true")
        self.addCleanup(tmp_dir.cleanup)
        rss = """<?xml version="1.0" encoding="UTF-8" ?>
<rss version="2.0">
  <channel>
    <item>
      <title>Newer learning headline</title>
      <link>https://news.google.com/rss/articles/newer</link>
      <description>Newer summary text.</description>
      <pubDate>Sun, 31 May 2026 11:00:00 GMT</pubDate>
      <source url="https://example.com">Example News</source>
    </item>
    <item>
      <title>Older learning headline</title>
      <link>https://example.com/older</link>
      <description>Older summary text.</description>
      <pubDate>Sun, 31 May 2026 10:00:00 GMT</pubDate>
      <source url="https://example.com">Example News</source>
    </item>
  </channel>
</rss>
"""
        rss_response = MagicMock(text=rss)
        resolve_response = MagicMock(url="https://example.com/newer")

        with patch(
            "news_fetcher.requests.get",
            side_effect=[rss_response, resolve_response],
        ) as mock_get:
            with patch("news_fetcher.datetime") as mock_datetime:
                mock_datetime.now.return_value = datetime(
                    2026, 5, 31, 12, 0, tzinfo=timezone.utc
                )
                news_item = fetch_latest_news("learning", config)

        self.assertIsNotNone(news_item)
        self.assertEqual(news_item.title, "Newer learning headline")
        self.assertEqual(news_item.link, "https://example.com/newer")
        request_url = mock_get.call_args.args[0]
        first_request_url = mock_get.call_args_list[0].args[0]
        self.assertIn("q=learning", first_request_url)
        self.assertIn("hl=en-US", first_request_url)
        self.assertEqual(request_url, "https://news.google.com/rss/articles/newer")
        rss_response.raise_for_status.assert_called_once()
        resolve_response.raise_for_status.assert_called_once()


class LoggerTests(unittest.TestCase):
    def test_build_tweet_log_entry_is_markdown_with_slot_marker(self) -> None:
        entry = build_tweet_log_entry(
            topic="coffee",
            tone="witty",
            tweet_text="Coffee is back.",
            time_taken_seconds=12.34,
            attempts=2,
            tweet_url="https://x.com/example/status/1",
            run_slot="12:00",
            timestamp="2026-05-15 12:00:00 IST",
            run_date="2026-05-15",
        )

        self.assertIn("## Tweet posted", entry)
        self.assertIn("- Run slot: 12:00", entry)
        self.assertIn("> Coffee is back.", entry)
        self.assertIn("<!-- tweet-slot:2026-05-15:12:00 -->", entry)

    def test_append_tweet_log_writes_markdown_and_duplicate_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "tweet-history.md"

            append_tweet_log(
                log_file_path=log_path,
                topic="coffee",
                tone="witty",
                tweet_text="Coffee is back.",
                time_taken_seconds=12.34,
                attempts=2,
                tweet_url="https://x.com/example/status/1",
                run_slot="12:00",
                timestamp="2026-05-15 12:00:00 IST",
                run_date="2026-05-15",
            )

            content = log_path.read_text(encoding="utf-8")
            logged = has_logged_slot(
                log_path, run_date="2026-05-15", run_slot="12:00"
            )

        self.assertIn("# Tweet History", content)
        self.assertIn("Topic: coffee", content)
        self.assertTrue(logged)

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
            news_published_at="2026-05-31 10:00 UTC",
            news_url="https://example.com/ai-agents",
        )

        self.assertIn("News title: AI agents reshape support workflows", entry)
        self.assertIn("News source: Example News", entry)
        self.assertIn("News published: 2026-05-31 10:00 UTC", entry)
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
        self.assertIn("Tone: witty", summary)
        self.assertIn("Time taken: 12.34 seconds", summary)
        self.assertIn("Attempts: 2", summary)
        self.assertIn("Coffee is back.", summary)
        self.assertNotIn("News reference", summary)
        self.assertNotIn("Tweet URL", summary)
        self.assertNotIn("tweet-slot", summary)

    def test_build_telegram_summary_includes_news_reference_when_provided(self) -> None:
        summary = build_telegram_summary(
            topic="ai agents",
            tone="serious",
            tweet_text="AI agents are moving into support queues. 🤖 #botWrites https://example.com/ai-agents",
            time_taken_seconds=8.5,
            attempts=1,
            news_title="AI agents reshape support workflows",
            news_source="Example News",
            news_published_at="2026-05-31 10:00 UTC",
            news_url="https://example.com/ai-agents",
        )

        self.assertIn("News reference:", summary)
        self.assertIn("AI agents reshape support workflows (Example News)", summary)
        self.assertIn("Published: 2026-05-31 10:00 UTC", summary)
        self.assertIn(
            "AI agents are moving into support queues. 🤖 #botWrites https://example.com/ai-agents",
            summary,
        )
        self.assertEqual(summary.count("https://example.com/ai-agents"), 1)
        self.assertNotIn("Tweet URL", summary)
        self.assertNotIn("tweet-slot", summary)

    def test_build_failure_telegram_summary_includes_error_and_news_reference(self) -> None:
        summary = build_failure_telegram_summary(
            topic="ai agents",
            tone="serious",
            error_message="Could not generate a valid tweet after 5 attempts: too generic.",
            news_title="AI agents reshape support workflows",
            news_source="Example News",
            news_published_at="2026-05-31 10:00 UTC",
            news_url="https://example.com/ai-agents",
        )

        self.assertIn("Tweet bot failed", summary)
        self.assertIn("Topic: ai agents", summary)
        self.assertIn("Tone: serious", summary)
        self.assertIn("News reference:", summary)
        self.assertIn("AI agents reshape support workflows", summary)
        self.assertIn("https://example.com/ai-agents", summary)
        self.assertIn("Error:", summary)
        self.assertIn("too generic", summary)
        self.assertNotIn("Tweet text:", summary)


class PublisherTests(unittest.TestCase):
    def test_build_post_text_appends_hashtag_once(self) -> None:
        self.assertEqual(build_post_text("Fresh take 🚀"), "Fresh take 🚀 #botWrites")
        self.assertEqual(
            build_post_text("Fresh take 🚀 #botWrites"), "Fresh take 🚀 #botWrites"
        )

    def test_build_post_text_appends_news_url_after_hashtag(self) -> None:
        self.assertEqual(
            build_post_text("Fresh take 🚀", "https://example.com/news"),
            "Fresh take 🚀 #botWrites https://example.com/news",
        )

    def test_max_generated_text_chars_reserves_suffix_space(self) -> None:
        self.assertEqual(max_generated_text_chars(280), 269)
        self.assertEqual(
            max_generated_text_chars(280, "https://example.com/news"),
            245,
        )


class TweetGeneratorTests(unittest.TestCase):
    def test_run_once_exits_cleanly_when_config_load_fails(self) -> None:
        buffer = StringIO()

        with patch.object(
            tweet_generator,
            "load_config",
            side_effect=ValueError("Missing required environment variable: TOPICS"),
        ):
            with patch("sys.stdout", buffer):
                result = tweet_generator.run_once()

        self.assertEqual(result, 0)
        self.assertIn("Could not generate tweet:", buffer.getvalue())

    def test_run_once_sends_short_telegram_summary_after_success(self) -> None:
        buffer = StringIO()
        tmp_dir, config = load_temp_config(
            POST_TO_X="true",
            X_API_KEY="key",
            X_API_KEY_SECRET="secret",
            X_ACCESS_TOKEN="token",
            X_ACCESS_TOKEN_SECRET="token-secret",
            X_USERNAME="example",
            TELEGRAM_BOT_TOKEN="bot-token",
            TELEGRAM_CHAT_ID="12345",
        )
        self.addCleanup(tmp_dir.cleanup)
        published = MagicMock(url="https://x.com/example/status/1")

        with patch.object(tweet_generator, "load_config", return_value=config):
            with patch.object(tweet_generator, "build_client", return_value=object()):
                with patch.object(
                    tweet_generator,
                    "generate_valid_tweet",
                    return_value=("Coffee is back. ☕", 1.0, 2),
                ):
                    with patch.object(
                        tweet_generator, "post_tweet_to_x", return_value=published
                    ) as mock_post:
                        with patch.object(
                            tweet_generator, "send_telegram_message"
                        ) as mock_telegram:
                            with patch("sys.stdout", buffer):
                                result = tweet_generator.run_once()

        self.assertEqual(result, 0)
        telegram_text = mock_telegram.call_args.args[1]
        self.assertIn("Topic:", telegram_text)
        self.assertIn("Tone:", telegram_text)
        self.assertIn("Attempts: 2", telegram_text)
        self.assertIn("Coffee is back. ☕ #botWrites", telegram_text)
        self.assertNotIn("News reference", telegram_text)
        self.assertNotIn("Tweet URL", telegram_text)
        self.assertNotIn("tweet-slot", telegram_text)
        mock_post.assert_called_once_with(config, "Coffee is back. ☕", news_url=None)
        self.assertIn("Tweet posted and logged.", buffer.getvalue())

    def test_run_once_uses_rss_news_when_available(self) -> None:
        buffer = StringIO()
        tmp_dir, config = load_temp_config(
            NEWS_ENABLED="true",
            POST_TO_X="true",
            X_API_KEY="key",
            X_API_KEY_SECRET="secret",
            X_ACCESS_TOKEN="token",
            X_ACCESS_TOKEN_SECRET="token-secret",
            X_USERNAME="example",
            TELEGRAM_BOT_TOKEN="bot-token",
            TELEGRAM_CHAT_ID="12345",
        )
        self.addCleanup(tmp_dir.cleanup)
        news_item = NewsItem(
            title="AI agents reshape support workflows",
            source="Example News",
            published_at=datetime(2026, 5, 31, 10, 0, tzinfo=timezone.utc),
            link="https://example.com/ai-agents",
            summary="Companies are deploying agents to resolve support tickets.",
        )
        published = MagicMock(url="https://x.com/example/status/2")

        with patch.object(tweet_generator, "load_config", return_value=config):
            with patch.object(tweet_generator, "build_client", return_value=object()):
                with patch.object(
                    tweet_generator,
                    "fetch_latest_news",
                    return_value=news_item,
                ) as mock_fetch:
                    with patch.object(
                        tweet_generator,
                        "generate_valid_tweet",
                        return_value=("AI agents are moving into support queues. 🤖", 1.0, 1),
                    ) as mock_generate:
                        with patch.object(
                            tweet_generator, "post_tweet_to_x", return_value=published
                        ) as mock_post:
                            with patch.object(
                                tweet_generator, "send_telegram_message"
                            ) as mock_telegram:
                                with patch("sys.stdout", buffer):
                                    result = tweet_generator.run_once()

        self.assertEqual(result, 0)
        mock_fetch.assert_called_once()
        self.assertIs(mock_generate.call_args.args[4], news_item)
        mock_post.assert_called_once_with(
            config,
            "AI agents are moving into support queues. 🤖",
            news_url="https://example.com/ai-agents",
        )
        log_content = config.log_file_path.read_text(encoding="utf-8")
        self.assertIn("News title: AI agents reshape support workflows", log_content)
        self.assertIn("News URL: https://example.com/ai-agents", log_content)
        self.assertIn(
            "AI agents are moving into support queues. 🤖 #botWrites https://example.com/ai-agents",
            log_content,
        )
        telegram_text = mock_telegram.call_args.args[1]
        self.assertIn("Topic:", telegram_text)
        self.assertIn("Tone:", telegram_text)
        self.assertIn(
            "AI agents are moving into support queues. 🤖 #botWrites https://example.com/ai-agents",
            telegram_text,
        )
        self.assertIn("News reference:", telegram_text)
        self.assertIn("AI agents reshape support workflows", telegram_text)
        self.assertIn("Example News", telegram_text)
        self.assertIn("2026-05-31 10:00 UTC", telegram_text)
        self.assertEqual(telegram_text.count("https://example.com/ai-agents"), 1)
        self.assertIn("Using RSS news:", buffer.getvalue())

    def test_run_once_sends_failure_telegram_when_generation_fails(self) -> None:
        buffer = StringIO()
        tmp_dir, config = load_temp_config(
            NEWS_ENABLED="true",
            POST_TO_X="true",
            X_API_KEY="key",
            X_API_KEY_SECRET="secret",
            X_ACCESS_TOKEN="token",
            X_ACCESS_TOKEN_SECRET="token-secret",
            X_USERNAME="example",
            TELEGRAM_BOT_TOKEN="bot-token",
            TELEGRAM_CHAT_ID="12345",
        )
        self.addCleanup(tmp_dir.cleanup)
        news_item = NewsItem(
            title="AI agents reshape support workflows",
            source="Example News",
            published_at=datetime(2026, 5, 31, 10, 0, tzinfo=timezone.utc),
            link="https://example.com/ai-agents",
            summary="Companies are deploying agents to resolve support tickets.",
        )

        with patch.object(tweet_generator, "load_config", return_value=config):
            with patch.object(tweet_generator, "build_client", return_value=object()):
                with patch.object(tweet_generator.random, "choice", side_effect=["learning", "serious"]):
                    with patch.object(
                        tweet_generator,
                        "fetch_latest_news",
                        return_value=news_item,
                    ):
                        with patch.object(
                            tweet_generator,
                            "generate_valid_tweet",
                            side_effect=RuntimeError(
                                "Could not generate a valid tweet after 5 attempts: too generic."
                            ),
                        ):
                            with patch.object(
                                tweet_generator, "post_tweet_to_x"
                            ) as mock_post:
                                with patch.object(
                                    tweet_generator, "send_telegram_message"
                                ) as mock_telegram:
                                    with patch("sys.stdout", buffer):
                                        result = tweet_generator.run_once()

        self.assertEqual(result, 0)
        mock_post.assert_not_called()
        telegram_text = mock_telegram.call_args.args[1]
        self.assertIn("Tweet bot failed", telegram_text)
        self.assertIn("Topic: learning", telegram_text)
        self.assertIn("Tone: serious", telegram_text)
        self.assertIn("News reference:", telegram_text)
        self.assertIn("AI agents reshape support workflows", telegram_text)
        self.assertIn("too generic", telegram_text)
        self.assertFalse(config.log_file_path.exists())

    def test_run_once_sends_failure_telegram_when_x_posting_fails(self) -> None:
        buffer = StringIO()
        tmp_dir, config = load_temp_config(
            POST_TO_X="true",
            X_API_KEY="key",
            X_API_KEY_SECRET="secret",
            X_ACCESS_TOKEN="token",
            X_ACCESS_TOKEN_SECRET="token-secret",
            X_USERNAME="example",
            TELEGRAM_BOT_TOKEN="bot-token",
            TELEGRAM_CHAT_ID="12345",
        )
        self.addCleanup(tmp_dir.cleanup)

        with patch.object(tweet_generator, "load_config", return_value=config):
            with patch.object(tweet_generator, "build_client", return_value=object()):
                with patch.object(tweet_generator.random, "choice", side_effect=["coffee", "witty"]):
                    with patch.object(
                        tweet_generator,
                        "generate_valid_tweet",
                        return_value=("Coffee is back. ☕", 1.0, 1),
                    ):
                        with patch.object(
                            tweet_generator,
                            "post_tweet_to_x",
                            side_effect=RuntimeError("X API returned 401"),
                        ):
                            with patch.object(
                                tweet_generator, "send_telegram_message"
                            ) as mock_telegram:
                                with patch("sys.stdout", buffer):
                                    result = tweet_generator.run_once()

        self.assertEqual(result, 0)
        telegram_text = mock_telegram.call_args.args[1]
        self.assertIn("Tweet bot failed", telegram_text)
        self.assertIn("Topic: coffee", telegram_text)
        self.assertIn("Tone: witty", telegram_text)
        self.assertIn("X API returned 401", telegram_text)
        self.assertNotIn("Coffee is back.", telegram_text)
        self.assertFalse(config.log_file_path.exists())

    def test_run_once_stays_clean_when_failure_telegram_send_fails(self) -> None:
        buffer = StringIO()
        tmp_dir, config = load_temp_config(
            POST_TO_X="true",
            X_API_KEY="key",
            X_API_KEY_SECRET="secret",
            X_ACCESS_TOKEN="token",
            X_ACCESS_TOKEN_SECRET="token-secret",
            X_USERNAME="example",
            TELEGRAM_BOT_TOKEN="bot-token",
            TELEGRAM_CHAT_ID="12345",
        )
        self.addCleanup(tmp_dir.cleanup)

        with patch.object(tweet_generator, "load_config", return_value=config):
            with patch.object(tweet_generator, "build_client", return_value=object()):
                with patch.object(
                    tweet_generator,
                    "generate_valid_tweet",
                    side_effect=RuntimeError("Could not generate a valid tweet"),
                ):
                    with patch.object(
                        tweet_generator,
                        "send_telegram_message",
                        side_effect=RuntimeError("Telegram send failed"),
                    ):
                        with patch("sys.stdout", buffer):
                            result = tweet_generator.run_once()

        self.assertEqual(result, 0)
        self.assertIn("Warning: Telegram failure alert delivery failed:", buffer.getvalue())
        self.assertFalse(config.log_file_path.exists())

    def test_run_once_falls_back_when_rss_has_no_recent_news(self) -> None:
        buffer = StringIO()
        tmp_dir, config = load_temp_config(
            NEWS_ENABLED="true",
            POST_TO_X="true",
            X_API_KEY="key",
            X_API_KEY_SECRET="secret",
            X_ACCESS_TOKEN="token",
            X_ACCESS_TOKEN_SECRET="token-secret",
            X_USERNAME="example",
        )
        self.addCleanup(tmp_dir.cleanup)
        published = MagicMock(url="https://x.com/example/status/3")

        with patch.object(tweet_generator, "load_config", return_value=config):
            with patch.object(tweet_generator, "build_client", return_value=object()):
                with patch.object(tweet_generator, "fetch_latest_news", return_value=None):
                    with patch.object(
                        tweet_generator,
                        "generate_valid_tweet",
                        return_value=("Coffee is back. ☕", 1.0, 1),
                    ) as mock_generate:
                        with patch.object(
                            tweet_generator, "post_tweet_to_x", return_value=published
                        ) as mock_post:
                            with patch("sys.stdout", buffer):
                                result = tweet_generator.run_once()

        self.assertEqual(result, 0)
        self.assertIsNone(mock_generate.call_args.args[4])
        mock_post.assert_called_once_with(config, "Coffee is back. ☕", news_url=None)
        self.assertIn("Using generic topic prompt", buffer.getvalue())

    def test_run_once_falls_back_when_rss_lookup_fails(self) -> None:
        buffer = StringIO()
        tmp_dir, config = load_temp_config(
            NEWS_ENABLED="true",
            POST_TO_X="true",
            X_API_KEY="key",
            X_API_KEY_SECRET="secret",
            X_ACCESS_TOKEN="token",
            X_ACCESS_TOKEN_SECRET="token-secret",
            X_USERNAME="example",
        )
        self.addCleanup(tmp_dir.cleanup)
        published = MagicMock(url="https://x.com/example/status/4")

        with patch.object(tweet_generator, "load_config", return_value=config):
            with patch.object(tweet_generator, "build_client", return_value=object()):
                with patch.object(
                    tweet_generator,
                    "fetch_latest_news",
                    side_effect=RuntimeError("RSS timed out"),
                ):
                    with patch.object(
                        tweet_generator,
                        "generate_valid_tweet",
                        return_value=("Learning still rewards curiosity. 📚", 1.0, 1),
                    ) as mock_generate:
                        with patch.object(
                            tweet_generator, "post_tweet_to_x", return_value=published
                        ) as mock_post:
                            with patch("sys.stdout", buffer):
                                result = tweet_generator.run_once()

        self.assertEqual(result, 0)
        self.assertIsNone(mock_generate.call_args.args[4])
        mock_post.assert_called_once_with(
            config, "Learning still rewards curiosity. 📚", news_url=None
        )
        self.assertIn("Warning: RSS news lookup failed", buffer.getvalue())
        self.assertNotIn(
            "News title:", config.log_file_path.read_text(encoding="utf-8")
        )

    def test_run_once_keeps_success_when_telegram_send_fails(self) -> None:
        buffer = StringIO()
        tmp_dir, config = load_temp_config(
            POST_TO_X="true",
            X_API_KEY="key",
            X_API_KEY_SECRET="secret",
            X_ACCESS_TOKEN="token",
            X_ACCESS_TOKEN_SECRET="token-secret",
            X_USERNAME="example",
            TELEGRAM_BOT_TOKEN="bot-token",
            TELEGRAM_CHAT_ID="12345",
        )
        self.addCleanup(tmp_dir.cleanup)
        published = MagicMock(url="https://x.com/example/status/1")

        with patch.object(tweet_generator, "load_config", return_value=config):
            with patch.object(tweet_generator, "build_client", return_value=object()):
                with patch.object(
                    tweet_generator,
                    "generate_valid_tweet",
                    return_value=("Coffee is back. ☕", 1.0, 2),
                ):
                    with patch.object(
                        tweet_generator, "post_tweet_to_x", return_value=published
                    ):
                        with patch.object(
                            tweet_generator,
                            "send_telegram_message",
                            side_effect=RuntimeError("Telegram send failed: chat not found"),
                        ):
                            with patch("sys.stdout", buffer):
                                result = tweet_generator.run_once()

        self.assertEqual(result, 0)
        self.assertIn("Warning: Telegram delivery failed:", buffer.getvalue())

    def test_run_once_logs_clear_timeout_message(self) -> None:
        buffer = StringIO()
        tmp_dir, config = load_temp_config()
        self.addCleanup(tmp_dir.cleanup)

        with patch.object(tweet_generator, "load_config", return_value=config):
            with patch.object(tweet_generator, "build_client", return_value=object()):
                with patch.object(
                    tweet_generator,
                    "generate_valid_tweet",
                    side_effect=TimeoutError("The read operation timed out"),
                ):
                    with patch("sys.stdout", buffer):
                        result = tweet_generator.run_once()

        output = buffer.getvalue()
        self.assertEqual(result, 0)
        self.assertIn("Ollama request timed out after", output)
        self.assertIn(config.ollama_model, output)
        self.assertIn(config.ollama_host, output)


class TelegramSenderTests(unittest.TestCase):
    def test_send_telegram_message_accepts_success_response(self) -> None:
        tmp_dir, config = load_temp_config(
            TELEGRAM_BOT_TOKEN="bot-token",
            TELEGRAM_CHAT_ID="12345",
        )
        self.addCleanup(tmp_dir.cleanup)
        response = MagicMock(status_code=200)
        response.json.return_value = {"ok": True}

        with patch("telegram_sender.requests.post", return_value=response) as mock_post:
            send_telegram_message(config, "hello")

        mock_post.assert_called_once()

    def test_send_telegram_message_raises_clear_error(self) -> None:
        tmp_dir, config = load_temp_config(
            TELEGRAM_BOT_TOKEN="bot-token",
            TELEGRAM_CHAT_ID="12345",
        )
        self.addCleanup(tmp_dir.cleanup)
        response = MagicMock(status_code=400)
        response.json.return_value = {"ok": False, "description": "chat not found"}

        with patch("telegram_sender.requests.post", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "Telegram send failed: chat not found"):
                send_telegram_message(config, "hello")


if __name__ == "__main__":
    unittest.main()
