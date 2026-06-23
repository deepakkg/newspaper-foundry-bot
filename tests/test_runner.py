from __future__ import annotations

import unittest
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import requests
from openai import OpenAIError

import notifications
import tweet_generator
from discord_approval import ApprovalDecision
from news_fetcher import NewsItem
from support import load_temp_config


def sample_news() -> NewsItem:
    return NewsItem(
        title="AI agents reshape support workflows",
        source="Example News",
        published_at=datetime(2026, 5, 31, 10, 0, tzinfo=timezone.utc),
        link="https://example.com/ai-agents",
        summary="Companies are deploying agents to resolve support tickets.",
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
        self.assertIn("Could not generate post:", buffer.getvalue())

    def test_run_once_approved_bluesky_and_x_publish_after_approval(self) -> None:
        buffer = StringIO()
        tmp_dir, config = load_temp_config(
            NEWS_ENABLED="true",
            POST_TO_BLUESKY="true",
            BLUESKY_HANDLE="example.bsky.social",
            BLUESKY_APP_PASSWORD="app-password",
            POST_TO_X="true",
            X_API_KEY="key",
            X_API_KEY_SECRET="secret",
            X_ACCESS_TOKEN="token",
            X_ACCESS_TOKEN_SECRET="token-secret",
            X_USERNAME="example",
        )
        self.addCleanup(tmp_dir.cleanup)
        order: list[str] = []

        def approve(*_args, **_kwargs):
            order.append("approval")
            return ApprovalDecision(status="approved", user_id="111", username="Deepak")

        def bluesky(*_args, **_kwargs):
            order.append("bluesky")
            return MagicMock(
                url="https://bsky.app/profile/example.bsky.social/post/abc",
                uri="at://did/post/abc",
            )

        def x_post(*_args, **_kwargs):
            order.append("x")
            return MagicMock(tweet_id="123", url="https://x.com/example/status/123")

        with patch.object(tweet_generator, "load_config", return_value=config):
            with patch.object(tweet_generator, "build_client", return_value=object()):
                with patch.object(tweet_generator.random, "choice", side_effect=["ai agents", "witty"]):
                    with patch.object(tweet_generator, "fetch_latest_news", return_value=sample_news()):
                        with patch.object(
                            tweet_generator,
                            "generate_valid_tweet",
                            return_value=("AI agents are moving into support queues. 🤖", 1.0, 1),
                        ):
                            with patch.object(tweet_generator, "request_discord_approval", side_effect=approve):
                                with patch.object(tweet_generator, "post_to_bluesky", side_effect=bluesky) as mock_bluesky:
                                    with patch.object(tweet_generator, "post_tweet_to_x", side_effect=x_post) as mock_x:
                                        with patch.object(notifications, "send_telegram_message"):
                                            with patch("sys.stdout", buffer):
                                                result = tweet_generator.run_once()

        self.assertEqual(result, 0)
        self.assertEqual(order, ["approval", "bluesky", "x"])
        mock_bluesky.assert_called_once()
        mock_x.assert_called_once()
        log_content = config.log_file_path.read_text(encoding="utf-8")
        self.assertIn("## Post published", log_content)
        self.assertIn("- Bluesky: published", log_content)
        self.assertIn("- X: published", log_content)
        self.assertIn("Decision by: Deepak", log_content)
        self.assertIn(
            "AI agents are moving into support queues. 🤖 #botWrites https://example.com/ai-agents",
            log_content,
        )
        self.assertIn("Post published and logged.", buffer.getvalue())

    def test_run_once_declined_publishes_nowhere_and_logs_outcome(self) -> None:
        buffer = StringIO()
        tmp_dir, config = load_temp_config(
            POST_TO_BLUESKY="true",
            BLUESKY_HANDLE="example.bsky.social",
            BLUESKY_APP_PASSWORD="app-password",
        )
        self.addCleanup(tmp_dir.cleanup)

        with patch.object(tweet_generator, "load_config", return_value=config):
            with patch.object(tweet_generator, "build_client", return_value=object()):
                with patch.object(
                    tweet_generator,
                    "generate_valid_tweet",
                    return_value=("Coffee is back. ☕", 1.0, 1),
                ):
                    with patch.object(
                        tweet_generator,
                        "request_discord_approval",
                        return_value=ApprovalDecision(status="declined", user_id="111", username="Deepak"),
                    ):
                        with patch.object(tweet_generator, "post_to_bluesky") as mock_bluesky:
                            with patch("sys.stdout", buffer):
                                result = tweet_generator.run_once()

        self.assertEqual(result, 0)
        mock_bluesky.assert_not_called()
        log_content = config.log_file_path.read_text(encoding="utf-8")
        self.assertIn("## Post declined", log_content)
        self.assertIn("- Bluesky: not published | declined", log_content)
        self.assertIn("Post was not published.", buffer.getvalue())

    def test_run_once_expired_publishes_nowhere_and_logs_outcome(self) -> None:
        tmp_dir, config = load_temp_config(
            POST_TO_X="true",
            X_API_KEY="key",
            X_API_KEY_SECRET="secret",
            X_ACCESS_TOKEN="token",
            X_ACCESS_TOKEN_SECRET="token-secret",
            X_USERNAME="example",
        )
        self.addCleanup(tmp_dir.cleanup)

        with patch.object(tweet_generator, "load_config", return_value=config):
            with patch.object(tweet_generator, "build_client", return_value=object()):
                with patch.object(
                    tweet_generator,
                    "generate_valid_tweet",
                    return_value=("Coffee is back. ☕", 1.0, 1),
                ):
                    with patch.object(
                        tweet_generator,
                        "request_discord_approval",
                        return_value=ApprovalDecision(status="expired"),
                    ):
                        with patch.object(tweet_generator, "post_tweet_to_x") as mock_x:
                            result = tweet_generator.run_once()

        self.assertEqual(result, 0)
        mock_x.assert_not_called()
        self.assertIn("## Post expired", config.log_file_path.read_text(encoding="utf-8"))

    def test_run_once_approved_instagram_renders_uploads_and_publishes(self) -> None:
        tmp_dir, config = load_temp_config(
            NEWS_ENABLED="true",
            POST_TO_INSTAGRAM="true",
            INSTAGRAM_ACCOUNT_ID="1789",
            INSTAGRAM_ACCESS_TOKEN="ig-token",
            CLOUDINARY_CLOUD_NAME="cloud",
            CLOUDINARY_API_KEY="cloud-key",
            CLOUDINARY_API_SECRET="cloud-secret",
        )
        self.addCleanup(tmp_dir.cleanup)
        uploaded = SimpleNamespace(
            secure_url="https://res.cloudinary.com/demo/post.png",
            public_id="content-bot/post",
        )
        published = SimpleNamespace(media_id="179", url="https://instagram.com/p/abc")

        with patch.object(tweet_generator, "load_config", return_value=config):
            with patch.object(tweet_generator, "build_client", return_value=object()):
                with patch.object(tweet_generator.random, "choice", side_effect=["ai agents", "analysis"]):
                    with patch.object(tweet_generator, "fetch_latest_news", return_value=sample_news()):
                        with patch.object(
                            tweet_generator,
                            "generate_valid_tweet",
                            return_value=("AI agents need better handoffs. 🤖", 1.0, 1),
                        ):
                            with patch.object(
                                tweet_generator,
                                "generate_instagram_hashtags",
                                return_value=["#AI", "#SupportOps"],
                            ):
                                with patch.object(
                                    tweet_generator,
                                    "request_discord_approval",
                                    return_value=ApprovalDecision(status="approved", user_id="111", username="Deepak"),
                                ):
                                    with patch.object(tweet_generator, "render_instagram_image", return_value=Path("/tmp/post.png")) as mock_render:
                                        with patch.object(tweet_generator, "upload_image_to_cloudinary", return_value=uploaded) as mock_upload:
                                            with patch.object(tweet_generator, "publish_instagram_image", return_value=published) as mock_publish:
                                                result = tweet_generator.run_once()

        self.assertEqual(result, 0)
        mock_render.assert_called_once()
        self.assertEqual(
            mock_render.call_args.args[0],
            "AI agents need better handoffs. 🤖",
        )
        mock_upload.assert_called_once()
        mock_publish.assert_called_once()
        caption = mock_publish.call_args.kwargs["caption"]
        self.assertIn("AI agents reshape support workflows", caption)
        self.assertIn("#aiagents", caption.lower())
        self.assertTrue(caption.strip().endswith("#botWrites"))
        log_content = config.log_file_path.read_text(encoding="utf-8")
        self.assertIn("- Instagram: published", log_content)
        self.assertIn("Instagram caption:", log_content)

    def test_run_once_instagram_failure_reports_partial_publish(self) -> None:
        buffer = StringIO()
        tmp_dir, config = load_temp_config(
            POST_TO_BLUESKY="true",
            BLUESKY_HANDLE="example.bsky.social",
            BLUESKY_APP_PASSWORD="app-password",
            POST_TO_INSTAGRAM="true",
            INSTAGRAM_ACCOUNT_ID="1789",
            INSTAGRAM_ACCESS_TOKEN="ig-token",
            CLOUDINARY_CLOUD_NAME="cloud",
            CLOUDINARY_API_KEY="cloud-key",
            CLOUDINARY_API_SECRET="cloud-secret",
            TELEGRAM_NOTIFICATIONS_ENABLED="true",
            TELEGRAM_BOT_TOKEN="bot-token",
            TELEGRAM_CHAT_ID="12345",
            DISCORD_NOTIFICATIONS_ENABLED="true",
            DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/1/token",
        )
        self.addCleanup(tmp_dir.cleanup)

        with patch.object(tweet_generator, "load_config", return_value=config):
            with patch.object(tweet_generator, "build_client", return_value=object()):
                with patch.object(
                    tweet_generator,
                    "generate_valid_tweet",
                    return_value=("Coffee is back. ☕", 1.0, 1),
                ):
                    with patch.object(
                        tweet_generator,
                        "generate_instagram_hashtags",
                        return_value=["#Coffee"],
                    ):
                        with patch.object(
                            tweet_generator,
                            "request_discord_approval",
                            return_value=ApprovalDecision(status="approved", user_id="111", username="Deepak"),
                        ):
                            with patch.object(
                                tweet_generator,
                                "post_to_bluesky",
                                return_value=MagicMock(
                                    url="https://bsky.app/profile/example.bsky.social/post/abc",
                                    uri="at://did/post/abc",
                                ),
                            ):
                                with patch.object(tweet_generator, "render_instagram_image", return_value=Path("/tmp/post.png")):
                                    with patch.object(
                                        tweet_generator,
                                        "upload_image_to_cloudinary",
                                        return_value=SimpleNamespace(
                                            secure_url="https://res.cloudinary.com/demo/post.png",
                                            public_id="content-bot/post",
                                        ),
                                    ):
                                        with patch.object(
                                            tweet_generator,
                                            "publish_instagram_image",
                                            side_effect=RuntimeError("bad image url"),
                                        ):
                                            with patch.object(notifications, "send_telegram_message") as mock_telegram:
                                                with patch.object(notifications, "send_discord_embed") as mock_discord:
                                                    with patch("sys.stdout", buffer):
                                                        result = tweet_generator.run_once()

        self.assertEqual(result, 0)
        output = buffer.getvalue()
        self.assertIn("Bluesky: published", output)
        self.assertIn("Instagram: failed", output)
        self.assertIn(
            "Cloudinary URL: https://res.cloudinary.com/demo/post.png",
            output,
        )
        self.assertIn("Post partially published and logged.", output)
        telegram_text = mock_telegram.call_args.args[1]
        self.assertIn("Post partially published", telegram_text)
        self.assertIn("Instagram: failed", telegram_text)
        embed = mock_discord.call_args.args[1]
        self.assertEqual(embed["title"], "Post partially published")
        self.assertIn("Instagram: failed", str(embed))
        log_content = config.log_file_path.read_text(encoding="utf-8")
        self.assertIn("## Post partially published", log_content)
        self.assertIn("- Bluesky: published", log_content)
        self.assertIn("- Instagram: failed", log_content)

    def test_run_once_one_platform_failure_still_logs_success_for_other_platforms(self) -> None:
        tmp_dir, config = load_temp_config(
            POST_TO_BLUESKY="true",
            BLUESKY_HANDLE="example.bsky.social",
            BLUESKY_APP_PASSWORD="app-password",
            POST_TO_X="true",
            X_API_KEY="key",
            X_API_KEY_SECRET="secret",
            X_ACCESS_TOKEN="token",
            X_ACCESS_TOKEN_SECRET="token-secret",
            X_USERNAME="example",
        )
        self.addCleanup(tmp_dir.cleanup)

        with patch.object(tweet_generator, "load_config", return_value=config):
            with patch.object(tweet_generator, "build_client", return_value=object()):
                with patch.object(
                    tweet_generator,
                    "generate_valid_tweet",
                    return_value=("Coffee is back. ☕", 1.0, 1),
                ):
                    with patch.object(
                        tweet_generator,
                        "request_discord_approval",
                        return_value=ApprovalDecision(status="approved", user_id="111", username="Deepak"),
                    ):
                        with patch.object(tweet_generator, "post_to_bluesky", side_effect=RuntimeError("rate limited")):
                            with patch.object(
                                tweet_generator,
                                "post_tweet_to_x",
                                return_value=MagicMock(tweet_id="123", url="https://x.com/example/status/123"),
                            ):
                                with patch.object(notifications, "send_telegram_message") as mock_telegram:
                                    result = tweet_generator.run_once()

        self.assertEqual(result, 0)
        mock_telegram.assert_not_called()
        log_content = config.log_file_path.read_text(encoding="utf-8")
        self.assertIn("- Bluesky: failed | rate limited", log_content)
        self.assertIn("- X: published", log_content)

    def test_run_once_all_platform_failures_send_failure_notification(self) -> None:
        buffer = StringIO()
        tmp_dir, config = load_temp_config(
            POST_TO_BLUESKY="true",
            BLUESKY_HANDLE="example.bsky.social",
            BLUESKY_APP_PASSWORD="app-password",
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
                    return_value=("Coffee is back. ☕", 1.0, 1),
                ):
                    with patch.object(
                        tweet_generator,
                        "request_discord_approval",
                        return_value=ApprovalDecision(status="approved", user_id="111", username="Deepak"),
                    ):
                        with patch.object(tweet_generator, "post_to_bluesky", side_effect=RuntimeError("rate limited")):
                            with patch.object(notifications, "send_telegram_message") as mock_telegram:
                                with patch("sys.stdout", buffer):
                                    result = tweet_generator.run_once()

        self.assertEqual(result, 0)
        telegram_text = mock_telegram.call_args.args[1]
        self.assertIn("Content bot failed", telegram_text)
        self.assertIn("All enabled platforms failed", telegram_text)
        log_content = config.log_file_path.read_text(encoding="utf-8")
        self.assertIn("## Post publish failed", log_content)
        self.assertNotIn("## Post published", log_content)

    def test_run_once_manual_mode_still_sends_post_text_without_publishing(self) -> None:
        buffer = StringIO()
        tmp_dir, config = load_temp_config(
            POST_TO_X="false",
            POST_TO_BLUESKY="false",
            POST_TO_INSTAGRAM="false",
            DISCORD_NOTIFICATIONS_ENABLED="true",
            DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/1/token",
        )
        self.addCleanup(tmp_dir.cleanup)

        with patch.object(tweet_generator, "load_config", return_value=config):
            with patch.object(tweet_generator, "build_client", return_value=object()):
                with patch.object(
                    tweet_generator,
                    "generate_valid_tweet",
                    return_value=("Learning still rewards curiosity. 📚", 1.0, 1),
                ):
                    with patch.object(notifications, "send_discord_message") as mock_message:
                        with patch.object(notifications, "send_discord_embed"):
                            with patch("sys.stdout", buffer):
                                result = tweet_generator.run_once()

        self.assertEqual(result, 0)
        mock_message.assert_called_once_with(
            config,
            "Learning still rewards curiosity. 📚 #botWrites",
        )
        self.assertFalse(config.log_file_path.exists())
        self.assertIn("Post ready for manual publishing.", buffer.getvalue())

    def test_run_once_sends_failure_telegram_when_generation_fails(self) -> None:
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
                with patch.object(tweet_generator.random, "choice", side_effect=["learning", "serious"]):
                    with patch.object(
                        tweet_generator,
                        "generate_valid_tweet",
                        side_effect=RuntimeError("Could not generate a valid post"),
                    ):
                        with patch.object(notifications, "send_telegram_message") as mock_telegram:
                            with patch("sys.stdout", buffer):
                                result = tweet_generator.run_once()

        self.assertEqual(result, 0)
        telegram_text = mock_telegram.call_args.args[1]
        self.assertIn("Content bot failed", telegram_text)
        self.assertIn("Could not generate a valid post", telegram_text)
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
