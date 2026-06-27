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

        self.assertEqual(config.llm_base_url, "http://localhost:11434/v1")
        self.assertEqual(config.llm_model, "gemma3:1b")
        self.assertIsNone(config.llm_api_key)
        self.assertTrue(config.post_to_x)
        self.assertFalse(config.news_enabled)
        self.assertEqual(config.news_recency_hours, 48)
        self.assertEqual(config.news_region, "US")
        self.assertEqual(config.news_language, "en")
        self.assertEqual(config.log_file_path, log_path)
        self.assertFalse(config.telegram_notifications_enabled)
        self.assertFalse(config.discord_notifications_enabled)

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

    def test_load_config_accepts_notification_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            write_env_file(
                env_path,
                TELEGRAM_NOTIFICATIONS_ENABLED="true",
                DISCORD_NOTIFICATIONS_ENABLED="true",
                DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/1/token",
            )

            config = load_config(env_path)

        self.assertTrue(config.telegram_notifications_enabled)
        self.assertTrue(config.discord_notifications_enabled)
        self.assertEqual(
            config.discord_webhook_url,
            "https://discord.com/api/webhooks/1/token",
        )

    def test_load_config_accepts_bluesky_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            write_env_file(
                env_path,
                POST_TO_BLUESKY="true",
                BLUESKY_HANDLE="example.bsky.social",
                BLUESKY_APP_PASSWORD="app-password",
                BLUESKY_SERVICE_URL="https://example-pds.com/",
                POST_TO_X="false",
            )

            config = load_config(env_path)

        self.assertTrue(config.post_to_bluesky)
        self.assertEqual(config.bluesky_handle, "example.bsky.social")
        self.assertEqual(config.bluesky_app_password, "app-password")
        self.assertEqual(config.bluesky_service_url, "https://example-pds.com")

    def test_load_config_requires_bluesky_credentials_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            write_env_file(env_path, POST_TO_BLUESKY="true")

            with self.assertRaisesRegex(
                ValueError,
                "POST_TO_BLUESKY is enabled but required Bluesky credentials "
                "are missing: BLUESKY_HANDLE, BLUESKY_APP_PASSWORD",
            ):
                load_config(env_path)

    def test_load_config_requires_x_credentials_when_x_enabled_with_bluesky(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            write_env_file(
                env_path,
                POST_TO_BLUESKY="true",
                BLUESKY_HANDLE="example.bsky.social",
                BLUESKY_APP_PASSWORD="app-password",
                POST_TO_X="true",
            )

            with self.assertRaisesRegex(
                ValueError,
                "POST_TO_X is enabled but required X credentials are missing",
            ):
                load_config(env_path)

    def test_load_config_accepts_approval_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            write_env_file(
                env_path,
                APPROVAL_TIMEOUT_MINUTES="45",
                DISCORD_BOT_TOKEN="bot-token",
                DISCORD_CHANNEL_ID="12345",
                DISCORD_APPROVER_USER_IDS="111,222",
            )

            config = load_config(env_path)

        self.assertEqual(config.approval_timeout_minutes, 45)
        self.assertEqual(config.discord_bot_token, "bot-token")
        self.assertEqual(config.discord_channel_id, "12345")
        self.assertEqual(config.discord_approver_user_ids, ["111", "222"])
        self.assertTrue(config.approval_required)

    def test_load_config_defaults_approval_required_to_true(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "LLM_BASE_URL=http://localhost:11434/v1",
                        "LLM_MODEL=gemma3:1b",
                        "TOPICS=coffee",
                        "TONES=witty",
                        "POST_TO_X=false",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.dict("os.environ", {}, clear=True):
                config = load_config(env_path)

        self.assertTrue(config.approval_required)

    def test_load_config_accepts_disabled_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            write_env_file(
                env_path,
                APPROVAL_REQUIRED="false",
            )

            config = load_config(env_path)

        self.assertFalse(config.approval_required)

    def test_load_config_requires_approval_settings_when_publishing_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            write_env_file(
                env_path,
                POST_TO_BLUESKY="true",
                BLUESKY_HANDLE="example.bsky.social",
                BLUESKY_APP_PASSWORD="app-password",
                DISCORD_BOT_TOKEN="",
                DISCORD_CHANNEL_ID="",
                DISCORD_APPROVER_USER_IDS="",
            )

            with self.assertRaisesRegex(
                ValueError,
                "required Discord approval settings are missing",
            ):
                load_config(env_path)

    def test_load_config_does_not_require_approval_settings_when_approval_disabled(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            write_env_file(
                env_path,
                POST_TO_BLUESKY="true",
                BLUESKY_HANDLE="example.bsky.social",
                BLUESKY_APP_PASSWORD="app-password",
                APPROVAL_REQUIRED="false",
                DISCORD_BOT_TOKEN="",
                DISCORD_CHANNEL_ID="",
                DISCORD_APPROVER_USER_IDS="",
            )

            config = load_config(env_path)

        self.assertTrue(config.post_to_bluesky)
        self.assertFalse(config.approval_required)
        self.assertIsNone(config.discord_bot_token)
        self.assertIsNone(config.discord_channel_id)
        self.assertEqual(config.discord_approver_user_ids, [])

    def test_load_config_accepts_instagram_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            write_env_file(
                env_path,
                POST_TO_INSTAGRAM="true",
                INSTAGRAM_ACCOUNT_ID="1789",
                INSTAGRAM_ACCESS_TOKEN="ig-token",
                INSTAGRAM_GRAPH_API_VERSION="v23.0",
                CLOUDINARY_CLOUD_NAME="cloud",
                CLOUDINARY_API_KEY="cloud-key",
                CLOUDINARY_API_SECRET="cloud-secret",
                CLOUDINARY_FOLDER="content-bot",
            )

            config = load_config(env_path)

        self.assertTrue(config.post_to_instagram)
        self.assertEqual(config.instagram_account_id, "1789")
        self.assertEqual(config.instagram_access_token, "ig-token")
        self.assertEqual(config.instagram_graph_base_url, "https://graph.instagram.com")
        self.assertEqual(config.cloudinary_cloud_name, "cloud")
        self.assertEqual(config.cloudinary_folder, "content-bot")

    def test_load_config_accepts_custom_instagram_graph_base_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            write_env_file(
                env_path,
                INSTAGRAM_GRAPH_BASE_URL="https://graph.facebook.com/",
            )

            config = load_config(env_path)

        self.assertEqual(config.instagram_graph_base_url, "https://graph.facebook.com")

    def test_load_config_accepts_article_link_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "history" / "tweet-history.md"
            env_path = Path(tmp_dir) / ".env"
            write_env_file(
                env_path,
                LOG_FILE_PATH=str(log_path),
                ARTICLE_LINKS_ENABLED="true",
                ARTICLE_LINKS_PAGE_URL="https://example.github.io/bot/article-links/",
                ARTICLE_LINKS_MAX_ITEMS="10",
            )

            config = load_config(env_path)

        self.assertTrue(config.article_links_enabled)
        self.assertEqual(
            config.article_links_page_url,
            "https://example.github.io/bot/article-links/",
        )
        self.assertEqual(config.article_links_max_items, 10)
        self.assertEqual(
            config.article_links_data_path,
            log_path.parent / "article-links" / "links.json",
        )
        self.assertEqual(
            config.article_links_html_path,
            log_path.parent / "article-links" / "index.html",
        )

    def test_load_config_defaults_article_links_to_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            write_env_file(env_path)

            config = load_config(env_path)

        self.assertFalse(config.article_links_enabled)
        self.assertIsNone(config.article_links_page_url)
        self.assertEqual(config.article_links_max_items, 25)

    def test_load_config_requires_instagram_and_cloudinary_credentials_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            write_env_file(env_path, POST_TO_INSTAGRAM="true")

            with self.assertRaisesRegex(
                ValueError,
                "POST_TO_INSTAGRAM is enabled but required Instagram/Cloudinary "
                "credentials are missing",
            ):
                load_config(env_path)

    def test_load_config_allows_missing_notification_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            write_env_file(
                env_path,
                TELEGRAM_NOTIFICATIONS_ENABLED="true",
                DISCORD_NOTIFICATIONS_ENABLED="true",
            )

            config = load_config(env_path)

        self.assertTrue(config.telegram_notifications_enabled)
        self.assertIsNone(config.telegram_bot_token)
        self.assertIsNone(config.telegram_chat_id)
        self.assertTrue(config.discord_notifications_enabled)
        self.assertIsNone(config.discord_webhook_url)

    def test_load_config_requires_llm_base_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            write_env_file(env_path, LLM_BASE_URL="")

            with self.assertRaisesRegex(
                ValueError, "Missing required environment variable: LLM_BASE_URL"
            ):
                load_config(env_path)

    def test_load_config_requires_llm_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            write_env_file(env_path, LLM_MODEL="")

            with self.assertRaisesRegex(
                ValueError, "Missing required environment variable: LLM_MODEL"
            ):
                load_config(env_path)

    def test_load_config_rejects_ollama_website_base_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            write_env_file(
                env_path,
                LLM_BASE_URL="https://ollama.com/",
                LLM_API_KEY="token",
            )

            with self.assertRaisesRegex(
                ValueError,
                "https://ollama.com, which is the Ollama website",
            ):
                load_config(env_path)

    def test_load_config_allows_missing_llm_api_key_for_local_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            write_env_file(
                env_path,
                LLM_BASE_URL="http://localhost:11434/v1",
                LLM_API_KEY="",
            )

            config = load_config(env_path)

        self.assertIsNone(config.llm_api_key)

    def test_load_config_requires_llm_api_key_for_hosted_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            write_env_file(
                env_path,
                LLM_BASE_URL="https://api.openai.com/v1",
                LLM_API_KEY="",
            )

            with self.assertRaisesRegex(
                ValueError, "LLM_API_KEY is required when using a hosted LLM endpoint"
            ):
                load_config(env_path)

    def test_load_config_reads_llm_timeout_seconds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            write_env_file(env_path, LLM_TIMEOUT_SECONDS="45")

            config = load_config(env_path)

        self.assertEqual(config.timeout_seconds, 45)

    def test_old_ollama_variables_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "OLLAMA_HOST=http://localhost:11434",
                        "OLLAMA_MODEL=gemma3:1b",
                        "OLLAMA_API_KEY=",
                        "OLLAMA_TIMEOUT_SECONDS=10",
                        "TOPICS=coffee",
                        "TONES=witty",
                        "POST_TO_X=false",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.dict("os.environ", {}, clear=True):
                with self.assertRaisesRegex(
                    ValueError, "Missing required environment variable: LLM_BASE_URL"
                ):
                    load_config(env_path)
