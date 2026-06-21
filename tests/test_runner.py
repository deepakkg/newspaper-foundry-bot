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
        self.assertIn("Could not generate post:", buffer.getvalue())

    def test_run_once_sends_short_telegram_summary_after_success(self) -> None:
        buffer = StringIO()
        tmp_dir, config = load_temp_config(
            POST_TO_X="true",
            X_API_KEY="key",
            X_API_KEY_SECRET="secret",
            X_ACCESS_TOKEN="token",
            X_ACCESS_TOKEN_SECRET="token-secret",
            X_USERNAME="example",
            TELEGRAM_NOTIFICATIONS_ENABLED="true",
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
                            notifications, "send_telegram_message"
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
        self.assertNotIn("Post URL", telegram_text)
        mock_post.assert_called_once_with(config, "Coffee is back. ☕", news_url=None)
        self.assertIn("Post published and logged.", buffer.getvalue())

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
            TELEGRAM_NOTIFICATIONS_ENABLED="true",
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
                                notifications, "send_telegram_message"
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

    def test_run_once_bluesky_only_posts_and_logs(self) -> None:
        buffer = StringIO()
        tmp_dir, config = load_temp_config(
            NEWS_ENABLED="true",
            POST_TO_BLUESKY="true",
            BLUESKY_HANDLE="example.bsky.social",
            BLUESKY_APP_PASSWORD="app-password",
            POST_TO_X="false",
            TELEGRAM_NOTIFICATIONS_ENABLED="true",
            TELEGRAM_BOT_TOKEN="bot-token",
            TELEGRAM_CHAT_ID="12345",
        )
        self.addCleanup(tmp_dir.cleanup)
        published = MagicMock(url="https://bsky.app/profile/example.bsky.social/post/3k4duaz5vfs2b")
        news_item = NewsItem(
            title="AI agents reshape support workflows",
            source="Example News",
            published_at=datetime(2026, 5, 31, 10, 0, tzinfo=timezone.utc),
            link="https://example.com/ai-agents",
            summary="Companies are deploying agents to resolve support tickets.",
        )

        with patch.object(tweet_generator, "load_config", return_value=config):
            with patch.object(tweet_generator, "build_client", return_value=object()):
                with patch.object(tweet_generator.random, "choice", side_effect=["ai agents", "witty"]):
                    with patch.object(
                        tweet_generator,
                        "fetch_latest_news",
                        return_value=news_item,
                    ):
                        with patch.object(
                            tweet_generator,
                            "generate_valid_tweet",
                            return_value=("AI agents are moving into support queues. 🤖", 1.0, 2),
                        ):
                            with patch.object(
                                tweet_generator, "post_to_bluesky", return_value=published
                            ) as mock_bluesky:
                                with patch.object(
                                    tweet_generator, "post_tweet_to_x"
                                ) as mock_x:
                                    with patch.object(
                                        notifications, "send_telegram_message"
                                    ) as mock_telegram:
                                        with patch("sys.stdout", buffer):
                                            result = tweet_generator.run_once()

        self.assertEqual(result, 0)
        mock_bluesky.assert_called_once_with(
            config,
            "AI agents are moving into support queues. 🤖 #botWrites",
            news_url="https://example.com/ai-agents",
            news_title="AI agents reshape support workflows",
            news_summary="Companies are deploying agents to resolve support tickets.",
        )
        mock_x.assert_not_called()
        log_content = config.log_file_path.read_text(encoding="utf-8")
        self.assertIn(
            "Post URL: https://bsky.app/profile/example.bsky.social/post/3k4duaz5vfs2b",
            log_content,
        )
        self.assertIn(
            "AI agents are moving into support queues. 🤖 #botWrites https://example.com/ai-agents",
            log_content,
        )
        self.assertIn("Post published and logged.", buffer.getvalue())
        telegram_text = mock_telegram.call_args.args[1]
        self.assertIn(
            "AI agents are moving into support queues. 🤖 #botWrites https://example.com/ai-agents",
            telegram_text,
        )

    def test_run_once_bluesky_failure_sends_failure_notification_without_log(self) -> None:
        buffer = StringIO()
        tmp_dir, config = load_temp_config(
            POST_TO_BLUESKY="true",
            BLUESKY_HANDLE="example.bsky.social",
            BLUESKY_APP_PASSWORD="app-password",
            POST_TO_X="false",
            TELEGRAM_NOTIFICATIONS_ENABLED="true",
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
                            "post_to_bluesky",
                            side_effect=RuntimeError("Bluesky posting failed: rate limited"),
                        ):
                            with patch.object(
                                tweet_generator, "post_tweet_to_x"
                            ) as mock_x:
                                with patch.object(
                                    notifications, "send_telegram_message"
                                ) as mock_telegram:
                                    with patch("sys.stdout", buffer):
                                        result = tweet_generator.run_once()

        self.assertEqual(result, 0)
        mock_x.assert_not_called()
        self.assertFalse(config.log_file_path.exists())
        telegram_text = mock_telegram.call_args.args[1]
        self.assertIn("Content bot failed", telegram_text)
        self.assertIn("Bluesky posting failed: rate limited", telegram_text)
        self.assertIn("Could not complete post run:", buffer.getvalue())

    def test_run_once_bluesky_enabled_does_not_enter_manual_mode(self) -> None:
        buffer = StringIO()
        tmp_dir, config = load_temp_config(
            POST_TO_BLUESKY="true",
            BLUESKY_HANDLE="example.bsky.social",
            BLUESKY_APP_PASSWORD="app-password",
            POST_TO_X="false",
            DISCORD_NOTIFICATIONS_ENABLED="true",
            DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/1/token",
        )
        self.addCleanup(tmp_dir.cleanup)
        published = MagicMock(url="https://bsky.app/profile/example.bsky.social/post/3k4duaz5vfs2b")

        with patch.object(tweet_generator, "load_config", return_value=config):
            with patch.object(tweet_generator, "build_client", return_value=object()):
                with patch.object(
                    tweet_generator,
                    "generate_valid_tweet",
                    return_value=("Coffee is back. ☕", 1.0, 1),
                ):
                    with patch.object(
                        tweet_generator, "post_to_bluesky", return_value=published
                    ):
                        with patch.object(
                            notifications, "send_discord_message"
                        ) as mock_manual_message:
                            with patch.object(notifications, "send_discord_embed"):
                                with patch("sys.stdout", buffer):
                                    result = tweet_generator.run_once()

        self.assertEqual(result, 0)
        mock_manual_message.assert_not_called()
        self.assertNotIn("Post ready for manual publishing.", buffer.getvalue())

    def test_run_once_manual_mode_sends_discord_embed_and_post_text(self) -> None:
        buffer = StringIO()
        tmp_dir, config = load_temp_config(
            NEWS_ENABLED="true",
            POST_TO_X="false",
            DISCORD_NOTIFICATIONS_ENABLED="true",
            DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/1/token",
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
                with patch.object(
                    tweet_generator,
                    "fetch_latest_news",
                    return_value=news_item,
                ):
                    with patch.object(
                        tweet_generator,
                        "generate_valid_tweet",
                        return_value=("AI agents are moving into support queues. 🤖", 1.0, 1),
                    ):
                        with patch.object(
                            tweet_generator, "post_tweet_to_x"
                        ) as mock_post:
                            with patch.object(
                                notifications, "send_discord_embed"
                            ) as mock_discord_embed:
                                with patch.object(
                                    notifications, "send_discord_message"
                                ) as mock_discord_message:
                                    with patch("sys.stdout", buffer):
                                        result = tweet_generator.run_once()

        self.assertEqual(result, 0)
        mock_post.assert_not_called()
        self.assertFalse(config.log_file_path.exists())
        embed = mock_discord_embed.call_args.args[1]
        self.assertEqual(embed["title"], "Post ready")
        self.assertIn(
            {"name": "News title", "value": "AI agents reshape support workflows", "inline": False},
            embed["fields"],
        )
        self.assertNotIn("Final post", [field["name"] for field in embed["fields"]])
        mock_discord_message.assert_called_once_with(
            config,
            "AI agents are moving into support queues. 🤖 #botWrites https://example.com/ai-agents",
        )
        self.assertIn("Post ready for manual publishing.", buffer.getvalue())

    def test_run_once_manual_mode_sends_telegram_success_summary(self) -> None:
        buffer = StringIO()
        tmp_dir, config = load_temp_config(
            POST_TO_X="false",
            TELEGRAM_NOTIFICATIONS_ENABLED="true",
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
                        return_value=("Coffee is back. ☕", 1.0, 2),
                    ):
                        with patch.object(
                            tweet_generator, "post_tweet_to_x"
                        ) as mock_post:
                            with patch.object(
                                notifications, "send_telegram_message"
                            ) as mock_telegram:
                                with patch("sys.stdout", buffer):
                                    result = tweet_generator.run_once()

        self.assertEqual(result, 0)
        mock_post.assert_not_called()
        self.assertFalse(config.log_file_path.exists())
        telegram_text = mock_telegram.call_args.args[1]
        self.assertIn("Topic: coffee", telegram_text)
        self.assertIn("Tone: witty", telegram_text)
        self.assertIn("Attempts: 2", telegram_text)
        self.assertIn("Coffee is back. ☕ #botWrites", telegram_text)
        self.assertNotIn("Content bot failed", telegram_text)

    def test_run_once_manual_mode_without_news_sends_post_text_without_link(self) -> None:
        buffer = StringIO()
        tmp_dir, config = load_temp_config(
            NEWS_ENABLED="true",
            POST_TO_X="false",
            DISCORD_NOTIFICATIONS_ENABLED="true",
            DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/1/token",
        )
        self.addCleanup(tmp_dir.cleanup)

        with patch.object(tweet_generator, "load_config", return_value=config):
            with patch.object(tweet_generator, "build_client", return_value=object()):
                with patch.object(tweet_generator, "fetch_latest_news", return_value=None):
                    with patch.object(
                        tweet_generator,
                        "generate_valid_tweet",
                        return_value=("Learning still rewards curiosity. 📚", 1.0, 1),
                    ):
                        with patch.object(
                            tweet_generator, "post_tweet_to_x"
                        ) as mock_post:
                            with patch.object(notifications, "send_discord_embed"):
                                with patch.object(
                                    notifications, "send_discord_message"
                                ) as mock_discord_message:
                                    with patch("sys.stdout", buffer):
                                        result = tweet_generator.run_once()

        self.assertEqual(result, 0)
        mock_post.assert_not_called()
        mock_discord_message.assert_called_once_with(
            config,
            "Learning still rewards curiosity. 📚 #botWrites",
        )
        self.assertFalse(config.log_file_path.exists())
        self.assertIn("Using generic topic prompt", buffer.getvalue())

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
            TELEGRAM_NOTIFICATIONS_ENABLED="true",
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
                                "Could not generate a valid post after 5 attempts: too generic."
                            ),
                        ):
                            with patch.object(
                                tweet_generator, "post_tweet_to_x"
                            ) as mock_post:
                                with patch.object(
                                    notifications, "send_telegram_message"
                                ) as mock_telegram:
                                    with patch("sys.stdout", buffer):
                                        result = tweet_generator.run_once()

        self.assertEqual(result, 0)
        mock_post.assert_not_called()
        telegram_text = mock_telegram.call_args.args[1]
        self.assertIn("Content bot failed", telegram_text)
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
            TELEGRAM_NOTIFICATIONS_ENABLED="true",
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
                                notifications, "send_telegram_message"
                            ) as mock_telegram:
                                with patch("sys.stdout", buffer):
                                    result = tweet_generator.run_once()

        self.assertEqual(result, 0)
        telegram_text = mock_telegram.call_args.args[1]
        self.assertIn("Content bot failed", telegram_text)
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
            TELEGRAM_NOTIFICATIONS_ENABLED="true",
            TELEGRAM_BOT_TOKEN="bot-token",
            TELEGRAM_CHAT_ID="12345",
        )
        self.addCleanup(tmp_dir.cleanup)

        with patch.object(tweet_generator, "load_config", return_value=config):
            with patch.object(tweet_generator, "build_client", return_value=object()):
                with patch.object(
                    tweet_generator,
                    "generate_valid_tweet",
                    side_effect=RuntimeError("Could not generate a valid post"),
                ):
                    with patch.object(
                        notifications,
                        "send_telegram_message",
                        side_effect=RuntimeError("Telegram send failed"),
                    ):
                        with patch("sys.stdout", buffer):
                            result = tweet_generator.run_once()

        self.assertEqual(result, 0)
        self.assertIn("Warning: Telegram failure alert delivery failed:", buffer.getvalue())
        self.assertFalse(config.log_file_path.exists())

    def test_run_once_stays_clean_when_failure_telegram_network_fails(self) -> None:
        buffer = StringIO()
        tmp_dir, config = load_temp_config(
            POST_TO_X="true",
            X_API_KEY="key",
            X_API_KEY_SECRET="secret",
            X_ACCESS_TOKEN="token",
            X_ACCESS_TOKEN_SECRET="token-secret",
            X_USERNAME="example",
            TELEGRAM_NOTIFICATIONS_ENABLED="true",
            TELEGRAM_BOT_TOKEN="bot-token",
            TELEGRAM_CHAT_ID="12345",
        )
        self.addCleanup(tmp_dir.cleanup)

        with patch.object(tweet_generator, "load_config", return_value=config):
            with patch.object(tweet_generator, "build_client", return_value=object()):
                with patch.object(
                    tweet_generator,
                    "generate_valid_tweet",
                    side_effect=RuntimeError("Could not generate a valid post"),
                ):
                    with patch.object(
                        notifications,
                        "send_telegram_message",
                        side_effect=requests.RequestException("Telegram timed out"),
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
            TELEGRAM_NOTIFICATIONS_ENABLED="true",
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
                            notifications,
                            "send_telegram_message",
                            side_effect=RuntimeError("Telegram send failed: chat not found"),
                        ):
                            with patch("sys.stdout", buffer):
                                result = tweet_generator.run_once()

        self.assertEqual(result, 0)
        self.assertIn("Warning: Telegram delivery failed:", buffer.getvalue())

    def test_run_once_keeps_success_when_telegram_network_fails(self) -> None:
        buffer = StringIO()
        tmp_dir, config = load_temp_config(
            POST_TO_X="true",
            X_API_KEY="key",
            X_API_KEY_SECRET="secret",
            X_ACCESS_TOKEN="token",
            X_ACCESS_TOKEN_SECRET="token-secret",
            X_USERNAME="example",
            TELEGRAM_NOTIFICATIONS_ENABLED="true",
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
                            notifications,
                            "send_telegram_message",
                            side_effect=requests.RequestException("Telegram timed out"),
                        ):
                            with patch("sys.stdout", buffer):
                                result = tweet_generator.run_once()

        self.assertEqual(result, 0)
        self.assertIn("Warning: Telegram delivery failed:", buffer.getvalue())
        mock_post.assert_called_once_with(
            config, "Coffee is back. ☕", news_url=None
        )
        self.assertIn(
            "Coffee is back. ☕ #botWrites",
            config.log_file_path.read_text(encoding="utf-8"),
        )

    def test_run_once_sends_discord_embed_after_success(self) -> None:
        buffer = StringIO()
        tmp_dir, config = load_temp_config(
            POST_TO_X="true",
            X_API_KEY="key",
            X_API_KEY_SECRET="secret",
            X_ACCESS_TOKEN="token",
            X_ACCESS_TOKEN_SECRET="token-secret",
            X_USERNAME="example",
            DISCORD_NOTIFICATIONS_ENABLED="true",
            DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/1/token",
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
                            notifications, "send_discord_embed"
                        ) as mock_discord:
                            with patch("sys.stdout", buffer):
                                result = tweet_generator.run_once()

        self.assertEqual(result, 0)
        embed = mock_discord.call_args.args[1]
        self.assertEqual(embed["title"], "Post published")
        self.assertIn(
            {"name": "Final post", "value": "Coffee is back. ☕ #botWrites", "inline": False},
            embed["fields"],
        )
        self.assertIn("Post published and logged.", buffer.getvalue())

    def test_run_once_sends_both_notification_channels(self) -> None:
        buffer = StringIO()
        tmp_dir, config = load_temp_config(
            POST_TO_X="true",
            X_API_KEY="key",
            X_API_KEY_SECRET="secret",
            X_ACCESS_TOKEN="token",
            X_ACCESS_TOKEN_SECRET="token-secret",
            X_USERNAME="example",
            TELEGRAM_NOTIFICATIONS_ENABLED="true",
            TELEGRAM_BOT_TOKEN="bot-token",
            TELEGRAM_CHAT_ID="12345",
            DISCORD_NOTIFICATIONS_ENABLED="true",
            DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/1/token",
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
                            notifications, "send_telegram_message"
                        ) as mock_telegram:
                            with patch.object(
                                notifications, "send_discord_embed"
                            ) as mock_discord:
                                with patch("sys.stdout", buffer):
                                    result = tweet_generator.run_once()

        self.assertEqual(result, 0)
        mock_telegram.assert_called_once()
        mock_discord.assert_called_once()

    def test_run_once_sends_discord_when_telegram_fails(self) -> None:
        buffer = StringIO()
        tmp_dir, config = load_temp_config(
            POST_TO_X="true",
            X_API_KEY="key",
            X_API_KEY_SECRET="secret",
            X_ACCESS_TOKEN="token",
            X_ACCESS_TOKEN_SECRET="token-secret",
            X_USERNAME="example",
            TELEGRAM_NOTIFICATIONS_ENABLED="true",
            TELEGRAM_BOT_TOKEN="bot-token",
            TELEGRAM_CHAT_ID="12345",
            DISCORD_NOTIFICATIONS_ENABLED="true",
            DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/1/token",
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
                            notifications,
                            "send_telegram_message",
                            side_effect=RuntimeError("Telegram failed"),
                        ):
                            with patch.object(
                                notifications, "send_discord_embed"
                            ) as mock_discord:
                                with patch("sys.stdout", buffer):
                                    result = tweet_generator.run_once()

        self.assertEqual(result, 0)
        mock_discord.assert_called_once()
        self.assertIn("Warning: Telegram delivery failed:", buffer.getvalue())

    def test_run_once_warns_when_enabled_credentials_are_missing(self) -> None:
        buffer = StringIO()
        tmp_dir, config = load_temp_config(
            POST_TO_X="true",
            X_API_KEY="key",
            X_API_KEY_SECRET="secret",
            X_ACCESS_TOKEN="token",
            X_ACCESS_TOKEN_SECRET="token-secret",
            X_USERNAME="example",
            TELEGRAM_NOTIFICATIONS_ENABLED="true",
            DISCORD_NOTIFICATIONS_ENABLED="true",
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
                        with patch("sys.stdout", buffer):
                            result = tweet_generator.run_once()

        self.assertEqual(result, 0)
        output = buffer.getvalue()
        self.assertIn("Warning: Telegram delivery failed:", output)
        self.assertIn("Warning: Discord delivery failed:", output)

    def test_run_once_sends_failure_discord_when_generation_fails(self) -> None:
        buffer = StringIO()
        tmp_dir, config = load_temp_config(
            POST_TO_X="true",
            X_API_KEY="key",
            X_API_KEY_SECRET="secret",
            X_ACCESS_TOKEN="token",
            X_ACCESS_TOKEN_SECRET="token-secret",
            X_USERNAME="example",
            DISCORD_NOTIFICATIONS_ENABLED="true",
            DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/1/token",
        )
        self.addCleanup(tmp_dir.cleanup)

        with patch.object(tweet_generator, "load_config", return_value=config):
            with patch.object(tweet_generator, "build_client", return_value=object()):
                with patch.object(tweet_generator.random, "choice", side_effect=["coffee", "witty"]):
                    with patch.object(
                        tweet_generator,
                        "generate_valid_tweet",
                        side_effect=RuntimeError("Could not generate a valid post"),
                    ):
                        with patch.object(
                            notifications, "send_discord_embed"
                        ) as mock_discord:
                            with patch("sys.stdout", buffer):
                                result = tweet_generator.run_once()

        self.assertEqual(result, 0)
        embed = mock_discord.call_args.args[1]
        self.assertEqual(embed["title"], "Content bot failed")
        self.assertIn(
            {"name": "Error", "value": "Could not generate a valid post", "inline": False},
            embed["fields"],
        )
        self.assertFalse(config.log_file_path.exists())

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
        self.assertIn("LLM request timed out after", output)
        self.assertIn(config.llm_model, output)
        self.assertIn(config.llm_base_url, output)

    def test_describe_failure_reports_llm_api_errors_generically(self) -> None:
        message = tweet_generator.describe_failure(OpenAIError("provider rejected request"))

        self.assertEqual(message, "LLM request failed: provider rejected request")

    def test_describe_failure_summarizes_html_provider_errors(self) -> None:
        message = tweet_generator.describe_failure(
            OpenAIError("<!doctype html><html><head><title>Ollama</title></head>")
        )

        self.assertEqual(
            message,
            "LLM request failed: provider returned an HTML page. "
            "Check that LLM_BASE_URL points to an OpenAI-compatible API "
            "endpoint, not a website.",
        )
