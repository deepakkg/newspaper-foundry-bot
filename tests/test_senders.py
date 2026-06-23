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
from discord_approval import ApprovalRequest, build_approval_embed, is_authorized_approver
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

class DiscordSenderTests(unittest.TestCase):
    def test_send_discord_embed_accepts_success_response(self) -> None:
        tmp_dir, config = load_temp_config(
            DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/1/token",
        )
        self.addCleanup(tmp_dir.cleanup)
        response = MagicMock(status_code=204, text="", reason="No Content")

        with patch("discord_sender.requests.post", return_value=response) as mock_post:
            send_discord_embed(config, {"title": "Post published", "fields": []})

        mock_post.assert_called_once_with(
            "https://discord.com/api/webhooks/1/token",
            json={
                "embeds": [{"title": "Post published", "fields": []}],
                "allowed_mentions": {"parse": []},
            },
            timeout=config.timeout_seconds,
        )

    def test_send_discord_embed_raises_clear_error(self) -> None:
        tmp_dir, config = load_temp_config(
            DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/1/token",
        )
        self.addCleanup(tmp_dir.cleanup)
        response = MagicMock(status_code=400, text="bad webhook", reason="Bad Request")

        with patch("discord_sender.requests.post", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "Discord send failed: bad webhook"):
                send_discord_embed(config, {"title": "Post published", "fields": []})

    def test_send_discord_message_accepts_success_response(self) -> None:
        tmp_dir, config = load_temp_config(
            DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/1/token",
        )
        self.addCleanup(tmp_dir.cleanup)
        response = MagicMock(status_code=204, text="", reason="No Content")

        with patch("discord_sender.requests.post", return_value=response) as mock_post:
            send_discord_message(config, "Fresh take 🚀 #botWrites")

        mock_post.assert_called_once_with(
            "https://discord.com/api/webhooks/1/token",
            json={
                "content": "Fresh take 🚀 #botWrites",
                "allowed_mentions": {"parse": []},
            },
            timeout=config.timeout_seconds,
        )

    def test_send_discord_message_raises_clear_error(self) -> None:
        tmp_dir, config = load_temp_config(
            DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/1/token",
        )
        self.addCleanup(tmp_dir.cleanup)
        response = MagicMock(status_code=400, text="bad webhook", reason="Bad Request")

        with patch("discord_sender.requests.post", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "Discord send failed: bad webhook"):
                send_discord_message(config, "Fresh take 🚀 #botWrites")

    def test_send_discord_embed_accepts_empty_success_response(self) -> None:
        tmp_dir, config = load_temp_config(
            DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/1/token",
        )
        self.addCleanup(tmp_dir.cleanup)
        response = MagicMock(status_code=200, text="", reason="OK")

        with patch("discord_sender.requests.post", return_value=response):
            send_discord_embed(config, {"title": "Post published", "fields": []})


class DiscordApprovalTests(unittest.TestCase):
    def test_is_authorized_approver_checks_configured_user_ids(self) -> None:
        tmp_dir, config = load_temp_config(DISCORD_APPROVER_USER_IDS="111,222")
        self.addCleanup(tmp_dir.cleanup)

        self.assertTrue(is_authorized_approver(config, "111"))
        self.assertTrue(is_authorized_approver(config, 222))
        self.assertFalse(is_authorized_approver(config, "333"))

    def test_build_approval_embed_includes_required_details(self) -> None:
        news_item = NewsItem(
            title="AI agents reshape support workflows",
            source="Example News",
            published_at=datetime(2026, 5, 31, 10, 0, tzinfo=timezone.utc),
            link="https://example.com/ai-agents",
            summary="Companies are deploying agents to resolve support tickets.",
        )

        embed = build_approval_embed(
            ApprovalRequest(
                topic="ai agents",
                tone="analysis",
                final_post_text="AI agents need better handoffs. 🤖 #botWrites https://example.com/ai-agents",
                instagram_caption="AI agents reshape support workflows\n\n#aiagents #botWrites",
                elapsed=4.2,
                attempts=2,
                target_platforms=["Bluesky", "Instagram"],
                news_item=news_item,
            )
        )

        self.assertEqual(embed["title"], "Post awaiting approval")
        field_names = [field["name"] for field in embed["fields"]]
        self.assertIn("Target platforms", field_names)
        self.assertIn("Article URL", field_names)
        self.assertIn("Final post", field_names)
        self.assertIn("Instagram caption preview", field_names)


if __name__ == "__main__":
    unittest.main()
