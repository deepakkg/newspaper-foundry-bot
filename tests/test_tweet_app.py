from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

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
    strip_html,
)
from publisher import build_post_text
from schedule_guard import decide_scheduled_run, resolve_current_slot
from telegram_sender import send_telegram_message
import tweet_generator


def write_env_file(path: Path, **overrides: str) -> None:
    values = {
        "OLLAMA_HOST": "http://localhost:11434",
        "TOPICS": "coffee,learning",
        "TONES": "witty,serious",
        "POST_TO_X": "false",
        "RUN_TIMEZONE": "Asia/Kolkata",
        "ENABLED_RUN_SLOTS": "06:00,12:00,18:00,22:00",
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

    def test_build_minimal_prompt_is_shorter_than_compact(self) -> None:
        compact_prompt = build_compact_prompt(
            "saas professional services", "serious", 230, 2
        )
        minimal_prompt = build_minimal_prompt(
            "saas professional services", "serious", 230
        )

        self.assertLess(len(minimal_prompt), len(compact_prompt))
        self.assertLess(len(minimal_prompt), 120)

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
        self.assertIn("Do not include the article URL.", prompt)

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
        tweet = "Narendra Modi keeps turning routine policy announcements into headline events, and that timing is half the story."

        result = validate_tweet(
            tweet,
            topic,
            topic_tokens,
            max_tweet_chars=230,
            attempt_number=1,
            max_retries=5,
        )

        self.assertIsNone(result)

    def test_rejects_generic_coffee_tweet(self) -> None:
        topic, topic_tokens = normalize_topic("coffee")
        tweet = "My morning coffee is definitely hitting the spot. Makes it easier to face the day."

        result = validate_tweet(
            tweet,
            topic,
            topic_tokens,
            max_tweet_chars=230,
            attempt_number=1,
            max_retries=5,
        )

        self.assertEqual(result, "too generic")


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
        self.assertEqual(config.run_timezone, "Asia/Kolkata")
        self.assertEqual(config.enabled_run_slots, ["06:00", "12:00", "18:00", "22:00"])
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

    def test_load_config_rejects_invalid_enabled_slot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            write_env_file(env_path, ENABLED_RUN_SLOTS="6:00")

            with self.assertRaisesRegex(
                ValueError, "ENABLED_RUN_SLOTS must be in HH:MM 24-hour format."
            ):
                load_config(env_path)

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

    def test_fetch_latest_news_returns_latest_usable_item(self) -> None:
        tmp_dir, config = load_temp_config(NEWS_ENABLED="true")
        self.addCleanup(tmp_dir.cleanup)
        rss = """<?xml version="1.0" encoding="UTF-8" ?>
<rss version="2.0">
  <channel>
    <item>
      <title>Newer learning headline</title>
      <link>https://example.com/newer</link>
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
        response = MagicMock(text=rss)

        with patch("news_fetcher.requests.get", return_value=response) as mock_get:
            with patch("news_fetcher.datetime") as mock_datetime:
                mock_datetime.now.return_value = datetime(
                    2026, 5, 31, 12, 0, tzinfo=timezone.utc
                )
                news_item = fetch_latest_news("learning", config)

        self.assertIsNotNone(news_item)
        self.assertEqual(news_item.title, "Newer learning headline")
        request_url = mock_get.call_args.args[0]
        self.assertIn("q=learning", request_url)
        self.assertIn("hl=en-US", request_url)
        response.raise_for_status.assert_called_once()


class ScheduleGuardTests(unittest.TestCase):
    def test_resolve_current_slot_accepts_delayed_workflow_start(self) -> None:
        now = datetime(2026, 5, 15, 12, 12, tzinfo=ZoneInfo("Asia/Kolkata"))

        self.assertEqual(resolve_current_slot(now), ("2026-05-15", "12:00"))

    def test_resolve_current_slot_accepts_later_retry_start(self) -> None:
        now = datetime(2026, 5, 15, 12, 59, tzinfo=ZoneInfo("Asia/Kolkata"))

        self.assertEqual(resolve_current_slot(now), ("2026-05-15", "12:00"))

    def test_resolve_current_slot_rejects_too_late_workflow_start(self) -> None:
        now = datetime(2026, 5, 15, 13, 16, tzinfo=ZoneInfo("Asia/Kolkata"))

        self.assertIsNone(resolve_current_slot(now))

    def test_decide_scheduled_run_runs_enabled_unlogged_slot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "tweet-history.md"
            env_path = Path(tmp_dir) / ".env"
            write_env_file(env_path, LOG_FILE_PATH=str(log_path))
            config = load_config(env_path)
            now = datetime(2026, 5, 15, 12, 0, tzinfo=ZoneInfo("Asia/Kolkata"))

            decision = decide_scheduled_run(config, now)

        self.assertTrue(decision.should_run)
        self.assertEqual(decision.run_date, "2026-05-15")
        self.assertEqual(decision.run_slot, "12:00")

    def test_decide_scheduled_run_skips_disabled_slot(self) -> None:
        tmp_dir, config = load_temp_config(ENABLED_RUN_SLOTS="06:00,18:00")
        self.addCleanup(tmp_dir.cleanup)
        now = datetime(2026, 5, 15, 12, 0, tzinfo=ZoneInfo("Asia/Kolkata"))

        decision = decide_scheduled_run(config, now)

        self.assertFalse(decision.should_run)
        self.assertIn("not enabled", decision.reason)

    def test_decide_scheduled_run_skips_already_logged_slot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "tweet-history.md"
            log_path.write_text(
                build_slot_marker(run_date="2026-05-15", run_slot="12:00"),
                encoding="utf-8",
            )
            env_path = Path(tmp_dir) / ".env"
            write_env_file(env_path, LOG_FILE_PATH=str(log_path))
            config = load_config(env_path)
            now = datetime(2026, 5, 15, 12, 0, tzinfo=ZoneInfo("Asia/Kolkata"))

            decision = decide_scheduled_run(config, now)

        self.assertFalse(decision.should_run)
        self.assertIn("already logged", decision.reason)


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
        self.assertNotIn("Tweet URL", summary)
        self.assertNotIn("tweet-slot", summary)


class PublisherTests(unittest.TestCase):
    def test_build_post_text_appends_hashtag_once(self) -> None:
        self.assertEqual(build_post_text("Fresh take"), "Fresh take #botWrites")
        self.assertEqual(
            build_post_text("Fresh take #botWrites"), "Fresh take #botWrites"
        )


class TweetGeneratorTests(unittest.TestCase):
    def test_run_once_skips_disabled_scheduled_slot(self) -> None:
        buffer = StringIO()
        tmp_dir, config = load_temp_config(ENABLED_RUN_SLOTS="06:00")
        self.addCleanup(tmp_dir.cleanup)
        now = datetime(2026, 5, 15, 12, 0, tzinfo=ZoneInfo("Asia/Kolkata"))

        with patch.object(tweet_generator, "load_config", return_value=config):
            with patch.object(tweet_generator, "build_client") as mock_client:
                with patch("sys.stdout", buffer):
                    result = tweet_generator.run_once(respect_schedule=True, now=now)

        self.assertEqual(result, 0)
        mock_client.assert_not_called()
        self.assertIn("not enabled", buffer.getvalue())

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
        now = datetime(2026, 5, 15, 12, 0, tzinfo=ZoneInfo("Asia/Kolkata"))

        with patch.object(tweet_generator, "load_config", return_value=config):
            with patch.object(tweet_generator, "build_client", return_value=object()):
                with patch.object(
                    tweet_generator,
                    "generate_valid_tweet",
                    return_value=("Coffee is back.", 1.0, 2),
                ):
                    with patch.object(
                        tweet_generator, "post_tweet_to_x", return_value=published
                    ):
                        with patch.object(
                            tweet_generator, "send_telegram_message"
                        ) as mock_telegram:
                            with patch("sys.stdout", buffer):
                                result = tweet_generator.run_once(
                                    respect_schedule=True, now=now
                                )

        self.assertEqual(result, 0)
        telegram_text = mock_telegram.call_args.args[1]
        self.assertIn("Topic:", telegram_text)
        self.assertIn("Tone:", telegram_text)
        self.assertIn("Attempts: 2", telegram_text)
        self.assertIn("Coffee is back.", telegram_text)
        self.assertNotIn("Tweet URL", telegram_text)
        self.assertNotIn("tweet-slot", telegram_text)
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
                        return_value=("AI agents are moving into support queues.", 1.0, 1),
                    ) as mock_generate:
                        with patch.object(
                            tweet_generator, "post_tweet_to_x", return_value=published
                        ):
                            with patch.object(
                                tweet_generator, "send_telegram_message"
                            ) as mock_telegram:
                                with patch("sys.stdout", buffer):
                                    result = tweet_generator.run_once()

        self.assertEqual(result, 0)
        mock_fetch.assert_called_once()
        self.assertIs(mock_generate.call_args.args[4], news_item)
        log_content = config.log_file_path.read_text(encoding="utf-8")
        self.assertIn("News title: AI agents reshape support workflows", log_content)
        self.assertIn("News URL: https://example.com/ai-agents", log_content)
        telegram_text = mock_telegram.call_args.args[1]
        self.assertIn("Topic:", telegram_text)
        self.assertIn("Tone:", telegram_text)
        self.assertIn("AI agents are moving into support queues.", telegram_text)
        self.assertNotIn("AI agents reshape support workflows", telegram_text)
        self.assertNotIn("https://example.com/ai-agents", telegram_text)
        self.assertIn("Using RSS news:", buffer.getvalue())

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
                        return_value=("Coffee is back.", 1.0, 1),
                    ) as mock_generate:
                        with patch.object(
                            tweet_generator, "post_tweet_to_x", return_value=published
                        ):
                            with patch("sys.stdout", buffer):
                                result = tweet_generator.run_once()

        self.assertEqual(result, 0)
        self.assertIsNone(mock_generate.call_args.args[4])
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
                        return_value=("Learning still rewards curiosity.", 1.0, 1),
                    ) as mock_generate:
                        with patch.object(
                            tweet_generator, "post_tweet_to_x", return_value=published
                        ):
                            with patch("sys.stdout", buffer):
                                result = tweet_generator.run_once()

        self.assertEqual(result, 0)
        self.assertIsNone(mock_generate.call_args.args[4])
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
                    return_value=("Coffee is back.", 1.0, 2),
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
        self.assertEqual(result, 1)
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
