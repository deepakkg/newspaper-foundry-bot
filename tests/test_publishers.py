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
import instagram_image
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
from cloudinary_uploader import upload_image_to_cloudinary
from instagram_content import (
    build_instagram_caption,
    extract_hashtags,
    fallback_hashtags,
    format_caption_news_title,
    generate_instagram_hashtags,
)
from instagram_image import (
    IMAGE_SIZE,
    build_instagram_image_body_text,
    extract_emojis,
    render_instagram_image,
)
from instagram_publisher import publish_instagram_image
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

    def test_build_post_text_without_url_appends_hashtag_only(self) -> None:
        self.assertEqual(
            build_post_text_without_url("Fresh take 🚀"),
            "Fresh take 🚀 #botWrites",
        )

    def test_max_generated_text_chars_reserves_suffix_space(self) -> None:
        self.assertEqual(max_generated_text_chars(280), 269)
        self.assertEqual(
            max_generated_text_chars(280, "https://example.com/news"),
            245,
        )


class InstagramContentTests(unittest.TestCase):
    def test_extract_hashtags_filters_duplicates_and_bot_tag(self) -> None:
        self.assertEqual(
            extract_hashtags("#AI #SupportOps #AI #botWrites not-a-tag"),
            ["#AI", "#SupportOps"],
        )

    def test_fallback_hashtags_include_topic_and_tone(self) -> None:
        self.assertEqual(
            fallback_hashtags("saas professional services", "analysis")[:2],
            ["#saasprofessionalservices", "#analysis"],
        )

    def test_build_instagram_caption_ends_with_botwrites(self) -> None:
        news_item = NewsItem(
            title="AI agents reshape support workflows",
            source="Example News",
            published_at=datetime(2026, 5, 31, 10, 0, tzinfo=timezone.utc),
            link="https://example.com/ai-agents",
            summary="Companies are deploying agents to resolve support tickets.",
        )

        caption = build_instagram_caption(
            topic="ai agents",
            tone="analysis",
            news_item=news_item,
            llm_hashtags=["#AI", "#SupportOps", "#botWrites"],
        )

        self.assertIn("News title: AI agents reshape support workflows", caption)
        self.assertIn("News source: Example News", caption)
        self.assertIn("News published: 2026-05-31 10:00 UTC", caption)
        self.assertIn("#aiagents #analysis #AI #SupportOps #botWrites", caption)
        self.assertNotIn("Hashtags:", caption)
        self.assertNotIn("Topic:", caption)
        self.assertNotIn("Tone:", caption)
        self.assertNotIn("https://example.com/ai-agents", caption)
        self.assertTrue(caption.strip().endswith("#botWrites"))

    def test_build_instagram_caption_uses_hashtags_only_without_news(self) -> None:
        caption = build_instagram_caption(
            topic="saas professional services",
            tone="analysis",
            news_item=None,
            llm_hashtags=["#SaaS", "#BusinessAnalysis"],
        )

        self.assertEqual(
            caption,
            "#saasprofessionalservices #analysis #SaaS #BusinessAnalysis #botWrites",
        )
        self.assertTrue(caption.strip().endswith("#botWrites"))

    def test_format_caption_news_title_removes_matching_source_suffix(self) -> None:
        self.assertEqual(
            format_caption_news_title(
                "CallMiner Enhances Real-Time Agent Performance, CX with New AI Capabilities - MarTech Cube",
                "MarTech Cube",
            ),
            "CallMiner Enhances Real-Time Agent Performance, CX with New AI Capabilities",
        )

    def test_format_caption_news_title_removes_dash_variants(self) -> None:
        self.assertEqual(
            format_caption_news_title("Fresh headline – Example News", "Example News"),
            "Fresh headline",
        )
        self.assertEqual(
            format_caption_news_title("Fresh headline — Example News", "Example News"),
            "Fresh headline",
        )

    def test_format_caption_news_title_preserves_non_matching_suffix(self) -> None:
        self.assertEqual(
            format_caption_news_title("Fresh headline - Other News", "Example News"),
            "Fresh headline - Other News",
        )

    def test_format_caption_news_title_preserves_source_in_middle(self) -> None:
        self.assertEqual(
            format_caption_news_title(
                "Example News explains why AI agents are changing support",
                "Example News",
            ),
            "Example News explains why AI agents are changing support",
        )

    def test_generate_instagram_hashtags_falls_back_on_invalid_llm_output(self) -> None:
        tmp_dir, config = load_temp_config()
        self.addCleanup(tmp_dir.cleanup)
        client = MagicMock()
        client.chat.completions.create.return_value = chat_response("no tags here")

        hashtags = generate_instagram_hashtags(
            client,
            config,
            "ai agents",
            "analysis",
            None,
        )

        self.assertIn("#aiagents", [tag.lower() for tag in hashtags])
        self.assertIn("#analysis", [tag.lower() for tag in hashtags])


class InstagramImageTests(unittest.TestCase):
    def test_build_instagram_image_body_text_removes_urls_hashtags_and_emojis(self) -> None:
        cleaned = build_instagram_image_body_text(
            "India loves an inquiry. 🚒📉 #botWrites https://www.bbc.com/news/story"
        )

        self.assertEqual(cleaned, "India loves an inquiry.")
        self.assertNotIn("🚒", cleaned)
        self.assertNotIn("#botWrites", cleaned)
        self.assertNotIn("https://", cleaned)

    def test_build_instagram_image_body_text_removes_emoji_leftovers(self) -> None:
        cleaned = build_instagram_image_body_text(
            "Cricket changed the game ⚙️\u200d\u20e3 □□ #botWrites"
        )

        self.assertEqual(cleaned, "Cricket changed the game.")
        self.assertNotIn("\ufe0f", cleaned)
        self.assertNotIn("\u200d", cleaned)
        self.assertNotIn("\u20e3", cleaned)
        self.assertNotIn("□", cleaned)

    def test_build_instagram_image_body_text_removes_keycap_emoji_without_stray_digit(
        self,
    ) -> None:
        cleaned = build_instagram_image_body_text("Top story 1️⃣ #botWrites")

        self.assertEqual(cleaned, "Top story.")

    def test_build_instagram_image_body_text_adds_period_when_missing(self) -> None:
        self.assertEqual(
            build_instagram_image_body_text("Cricket changed the game 🏏"),
            "Cricket changed the game.",
        )

    def test_build_instagram_image_body_text_preserves_sentence_punctuation(self) -> None:
        self.assertEqual(
            build_instagram_image_body_text("Cricket changed the game! 🏏"),
            "Cricket changed the game!",
        )

    def test_extract_emojis_reads_raw_llm_text(self) -> None:
        self.assertEqual(extract_emojis("Cricket changed the game 🏏📉"), "🏏📉")

    def test_wrap_text_does_not_leave_final_period_alone(self) -> None:
        font = instagram_image._load_font(58)
        from PIL import Image, ImageDraw

        draw = ImageDraw.Draw(Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE)))
        text = (
            "The stock market rewards those who look where others don't. A screaming "
            "buy value stock hiding in plain sight proves that obvious isn't always "
            "priced in. Patience beats the hype."
        )

        lines = instagram_image._wrap_text(draw, text, font, 820)

        self.assertNotEqual(lines[-1], ".")
        self.assertFalse(any(line.strip() in {".", "!", "?"} for line in lines))

    def test_wrap_text_does_not_leave_exclamation_or_question_alone(self) -> None:
        font = instagram_image._load_font(58)
        from PIL import Image, ImageDraw

        draw = ImageDraw.Draw(Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE)))

        for punctuation in ("!", "?"):
            lines = instagram_image._wrap_text(
                draw,
                f"Markets price the obvious slowly {punctuation}",
                font,
                610,
            )
            self.assertNotEqual(lines[-1], punctuation)
            self.assertFalse(any(line.strip() == punctuation for line in lines))

    def test_wrap_text_rebalances_tiny_final_line(self) -> None:
        font = instagram_image._load_font(58)
        from PIL import Image, ImageDraw

        draw = ImageDraw.Draw(Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE)))
        lines = instagram_image._wrap_text(
            draw,
            "Patience beats the hype.",
            font,
            690,
        )

        self.assertGreater(len(lines[-1].strip()), 2)

    def test_render_instagram_image_skips_emoji_line_when_unsupported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "post.png"
            with patch.object(instagram_image, "_load_emoji_font", return_value=None):
                render_instagram_image("Cricket changed the game 🏏", output_path)

            from PIL import Image

            with Image.open(output_path) as image:
                self.assertEqual(image.size, (1080, 1080))

    def test_render_instagram_image_draws_supported_emoji_as_separate_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "post.png"
            emoji_font = MagicMock()
            emoji_font.getbbox.return_value = (0, 0, 40, 40)
            drawn_text: list[tuple[str, bool]] = []

            def capture_centered_text(*args, **kwargs):
                drawn_text.append((args[1], kwargs.get("embedded_color", False)))

            with patch.object(instagram_image, "_load_emoji_font", return_value=emoji_font):
                with patch.object(instagram_image, "_emoji_text_renders_cleanly", return_value=True):
                    with patch.object(
                        instagram_image,
                        "_draw_centered_text",
                        side_effect=capture_centered_text,
                    ):
                        render_instagram_image("Cricket changed the game 🏏", output_path)

            self.assertIn(("🏏", True), drawn_text)
            self.assertIn(("#botWrites", False), drawn_text)

    def test_render_instagram_image_creates_square_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "post.png"
            render_instagram_image(
                "SaaS professional services should reduce churn, not pad margins. ⚙️ #botWrites",
                output_path,
            )

            from PIL import Image

            with Image.open(output_path) as image:
                self.assertEqual(image.size, (1080, 1080))
                self.assertEqual(image.format, "PNG")

    def test_render_instagram_image_handles_stock_market_regression_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "post.png"
            with patch.object(instagram_image, "_load_emoji_font", return_value=None):
                render_instagram_image(
                    "The stock market rewards those who look where others don't. "
                    "A screaming buy value stock hiding in plain sight proves that "
                    "obvious isn't always priced in. Patience beats the hype. 📉 #botWrites",
                    output_path,
                )

            from PIL import Image

            with Image.open(output_path) as image:
                self.assertEqual(image.size, (1080, 1080))


class CloudinaryUploaderTests(unittest.TestCase):
    def test_upload_image_to_cloudinary_returns_secure_url(self) -> None:
        tmp_dir, config = load_temp_config(
            POST_TO_INSTAGRAM="true",
            INSTAGRAM_ACCOUNT_ID="1789",
            INSTAGRAM_ACCESS_TOKEN="ig-token",
            CLOUDINARY_CLOUD_NAME="cloud",
            CLOUDINARY_API_KEY="cloud-key",
            CLOUDINARY_API_SECRET="cloud-secret",
        )
        self.addCleanup(tmp_dir.cleanup)

        with patch(
            "cloudinary.uploader.upload",
            return_value={
                "secure_url": "https://res.cloudinary.com/demo/post.png",
                "public_id": "content-bot/post",
            },
        ) as mock_upload:
            uploaded = upload_image_to_cloudinary(config, Path("/tmp/post.png"))

        self.assertEqual(uploaded.secure_url, "https://res.cloudinary.com/demo/post.png")
        self.assertEqual(uploaded.public_id, "content-bot/post")
        mock_upload.assert_called_once()


class InstagramPublisherTests(unittest.TestCase):
    def test_publish_instagram_image_creates_and_publishes_container(self) -> None:
        tmp_dir, config = load_temp_config(
            POST_TO_INSTAGRAM="true",
            INSTAGRAM_ACCOUNT_ID="1789",
            INSTAGRAM_ACCESS_TOKEN="ig-token",
            CLOUDINARY_CLOUD_NAME="cloud",
            CLOUDINARY_API_KEY="cloud-key",
            CLOUDINARY_API_SECRET="cloud-secret",
        )
        self.addCleanup(tmp_dir.cleanup)
        media_response = MagicMock(status_code=200)
        media_response.json.return_value = {"id": "container-1"}
        status_response = MagicMock(status_code=200)
        status_response.json.return_value = {"status_code": "FINISHED"}
        publish_response = MagicMock(status_code=200)
        publish_response.json.return_value = {
            "id": "media-1",
            "permalink": "https://instagram.com/p/abc",
        }

        with patch(
            "instagram_publisher.requests.post",
            side_effect=[media_response, publish_response],
        ) as mock_post:
            with patch("instagram_publisher.requests.get", return_value=status_response) as mock_get:
                published = publish_instagram_image(
                    config,
                    image_url="https://res.cloudinary.com/demo/post.png",
                    caption="Caption #botWrites",
                )

        self.assertEqual(published.media_id, "media-1")
        self.assertEqual(published.url, "https://instagram.com/p/abc")
        self.assertEqual(mock_post.call_count, 2)
        mock_get.assert_called_once()
        self.assertEqual(
            mock_get.call_args.args[0],
            "https://graph.instagram.com/v23.0/container-1",
        )
        self.assertEqual(
            mock_post.call_args_list[0].args[0],
            "https://graph.instagram.com/v23.0/1789/media",
        )
        self.assertEqual(
            mock_post.call_args_list[1].args[0],
            "https://graph.instagram.com/v23.0/1789/media_publish",
        )

    def test_publish_instagram_image_accepts_custom_graph_base_url(self) -> None:
        tmp_dir, config = load_temp_config(
            POST_TO_INSTAGRAM="true",
            INSTAGRAM_ACCOUNT_ID="1789",
            INSTAGRAM_ACCESS_TOKEN="ig-token",
            INSTAGRAM_GRAPH_BASE_URL="https://graph.facebook.com",
            CLOUDINARY_CLOUD_NAME="cloud",
            CLOUDINARY_API_KEY="cloud-key",
            CLOUDINARY_API_SECRET="cloud-secret",
        )
        self.addCleanup(tmp_dir.cleanup)
        media_response = MagicMock(status_code=200)
        media_response.json.return_value = {"id": "container-1"}
        status_response = MagicMock(status_code=200)
        status_response.json.return_value = {"status_code": "FINISHED"}
        publish_response = MagicMock(status_code=200)
        publish_response.json.return_value = {"id": "media-1"}

        with patch(
            "instagram_publisher.requests.post",
            side_effect=[media_response, publish_response],
        ) as mock_post:
            with patch("instagram_publisher.requests.get", return_value=status_response):
                publish_instagram_image(
                    config,
                    image_url="https://res.cloudinary.com/demo/post.png",
                    caption="Caption #botWrites",
                )

        self.assertEqual(
            mock_post.call_args_list[0].args[0],
            "https://graph.facebook.com/v23.0/1789/media",
        )
        self.assertEqual(
            mock_post.call_args_list[1].args[0],
            "https://graph.facebook.com/v23.0/1789/media_publish",
        )

    def test_publish_instagram_image_waits_for_media_container_to_finish(self) -> None:
        tmp_dir, config = load_temp_config(
            POST_TO_INSTAGRAM="true",
            INSTAGRAM_ACCOUNT_ID="1789",
            INSTAGRAM_ACCESS_TOKEN="ig-token",
            CLOUDINARY_CLOUD_NAME="cloud",
            CLOUDINARY_API_KEY="cloud-key",
            CLOUDINARY_API_SECRET="cloud-secret",
        )
        self.addCleanup(tmp_dir.cleanup)
        media_response = MagicMock(status_code=200)
        media_response.json.return_value = {"id": "container-1"}
        processing_response = MagicMock(status_code=200)
        processing_response.json.return_value = {"status_code": "IN_PROGRESS"}
        finished_response = MagicMock(status_code=200)
        finished_response.json.return_value = {"status_code": "FINISHED"}
        publish_response = MagicMock(status_code=200)
        publish_response.json.return_value = {"id": "media-1"}

        with patch(
            "instagram_publisher.requests.post",
            side_effect=[media_response, publish_response],
        ) as mock_post:
            with patch(
                "instagram_publisher.requests.get",
                side_effect=[processing_response, finished_response],
            ) as mock_get:
                with patch("instagram_publisher.time.sleep") as mock_sleep:
                    published = publish_instagram_image(
                        config,
                        image_url="https://res.cloudinary.com/demo/post.png",
                        caption="Caption #botWrites",
                    )

        self.assertEqual(published.media_id, "media-1")
        self.assertEqual(mock_get.call_count, 2)
        mock_sleep.assert_called_once_with(5)
        self.assertEqual(mock_post.call_count, 2)

    def test_publish_instagram_image_stops_when_media_container_fails(self) -> None:
        tmp_dir, config = load_temp_config(
            POST_TO_INSTAGRAM="true",
            INSTAGRAM_ACCOUNT_ID="1789",
            INSTAGRAM_ACCESS_TOKEN="ig-token",
            CLOUDINARY_CLOUD_NAME="cloud",
            CLOUDINARY_API_KEY="cloud-key",
            CLOUDINARY_API_SECRET="cloud-secret",
        )
        self.addCleanup(tmp_dir.cleanup)
        media_response = MagicMock(status_code=200)
        media_response.json.return_value = {"id": "container-1"}
        failed_response = MagicMock(status_code=200)
        failed_response.json.return_value = {
            "status_code": "ERROR",
            "status": "Image fetch failed",
        }

        with patch("instagram_publisher.requests.post", return_value=media_response) as mock_post:
            with patch("instagram_publisher.requests.get", return_value=failed_response):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Instagram media container processing failed",
                ):
                    publish_instagram_image(
                        config,
                        image_url="https://res.cloudinary.com/demo/post.png",
                        caption="Caption #botWrites",
                    )

        self.assertEqual(mock_post.call_count, 1)

    def test_publish_instagram_image_times_out_waiting_for_media_container(self) -> None:
        tmp_dir, config = load_temp_config(
            POST_TO_INSTAGRAM="true",
            INSTAGRAM_ACCOUNT_ID="1789",
            INSTAGRAM_ACCESS_TOKEN="ig-token",
            CLOUDINARY_CLOUD_NAME="cloud",
            CLOUDINARY_API_KEY="cloud-key",
            CLOUDINARY_API_SECRET="cloud-secret",
        )
        self.addCleanup(tmp_dir.cleanup)
        media_response = MagicMock(status_code=200)
        media_response.json.return_value = {"id": "container-1"}
        processing_response = MagicMock(status_code=200)
        processing_response.json.return_value = {"status_code": "IN_PROGRESS"}

        with patch("instagram_publisher.requests.post", return_value=media_response) as mock_post:
            with patch("instagram_publisher.requests.get", return_value=processing_response) as mock_get:
                with patch("instagram_publisher.time.sleep"):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "Instagram media container was not ready",
                    ):
                        publish_instagram_image(
                            config,
                            image_url="https://res.cloudinary.com/demo/post.png",
                            caption="Caption #botWrites",
                        )

        self.assertEqual(mock_post.call_count, 1)
        self.assertEqual(mock_get.call_count, 8)

    def test_publish_instagram_image_raises_clear_error(self) -> None:
        tmp_dir, config = load_temp_config(
            POST_TO_INSTAGRAM="true",
            INSTAGRAM_ACCOUNT_ID="1789",
            INSTAGRAM_ACCESS_TOKEN="ig-token",
            CLOUDINARY_CLOUD_NAME="cloud",
            CLOUDINARY_API_KEY="cloud-key",
            CLOUDINARY_API_SECRET="cloud-secret",
        )
        self.addCleanup(tmp_dir.cleanup)
        response = MagicMock(status_code=400)
        response.json.return_value = {"error": {"message": "bad image url"}}

        with patch("instagram_publisher.requests.post", return_value=response):
            with self.assertRaisesRegex(
                RuntimeError,
                "Instagram media container creation failed: bad image url",
            ):
                publish_instagram_image(
                    config,
                    image_url="https://example.com/image.png",
                    caption="Caption",
                )

    def test_publish_instagram_image_explains_invalid_token_error(self) -> None:
        tmp_dir, config = load_temp_config(
            POST_TO_INSTAGRAM="true",
            INSTAGRAM_ACCOUNT_ID="1789",
            INSTAGRAM_ACCESS_TOKEN="bad-token",
            CLOUDINARY_CLOUD_NAME="cloud",
            CLOUDINARY_API_KEY="cloud-key",
            CLOUDINARY_API_SECRET="cloud-secret",
        )
        self.addCleanup(tmp_dir.cleanup)
        response = MagicMock(status_code=400)
        response.json.return_value = {
            "error": {
                "message": "Invalid OAuth access token - Cannot parse access token"
            }
        }

        with patch("instagram_publisher.requests.post", return_value=response):
            with self.assertRaisesRegex(
                RuntimeError,
                "Instagram access token is invalid or malformed",
            ) as error:
                publish_instagram_image(
                    config,
                    image_url="https://example.com/image.png",
                    caption="Caption",
                )

        self.assertIn(
            "Original error: Invalid OAuth access token - Cannot parse access token",
            str(error.exception),
        )

class BlueskyPublisherTests(unittest.TestCase):
    def test_build_bluesky_post_url_uses_record_key(self) -> None:
        url = build_bluesky_post_url(
            "example.bsky.social",
            "at://did:plc:abc123/app.bsky.feed.post/3k4duaz5vfs2b",
        )

        self.assertEqual(
            url,
            "https://bsky.app/profile/example.bsky.social/post/3k4duaz5vfs2b",
        )

    def test_post_to_bluesky_without_news_url_sends_plain_text(self) -> None:
        tmp_dir, config = load_temp_config(
            POST_TO_BLUESKY="true",
            BLUESKY_HANDLE="example.bsky.social",
            BLUESKY_APP_PASSWORD="app-password",
            BLUESKY_SERVICE_URL="https://bsky.social",
        )
        self.addCleanup(tmp_dir.cleanup)
        client = MagicMock()
        client.send_post.return_value = SimpleNamespace(
            uri="at://did:plc:abc123/app.bsky.feed.post/3k4duaz5vfs2b",
            cid="bafyexample",
        )

        with patch("bluesky_publisher.Client", return_value=client) as mock_client:
            published = post_to_bluesky(config, "Fresh take 🚀 #botWrites")

        mock_client.assert_called_once_with(base_url="https://bsky.social")
        client.login.assert_called_once_with("example.bsky.social", "app-password")
        client.send_post.assert_called_once_with("Fresh take 🚀 #botWrites", embed=None)
        self.assertEqual(published.uri, "at://did:plc:abc123/app.bsky.feed.post/3k4duaz5vfs2b")
        self.assertEqual(published.cid, "bafyexample")
        self.assertEqual(
            published.url,
            "https://bsky.app/profile/example.bsky.social/post/3k4duaz5vfs2b",
        )

    def test_fetch_link_card_metadata_reads_open_graph_fields(self) -> None:
        response = MagicMock(
            url="https://example.com/story",
            text=(
                '<meta property="og:title" content="Publisher headline">'
                '<meta property="og:description" content="Publisher summary">'
                '<meta property="og:image" content="/image.jpg">'
            ),
        )

        with patch("link_preview.requests.get", return_value=response) as mock_get:
            metadata = fetch_link_card_metadata(
                "https://example.com/story",
                fallback_title="RSS headline",
                fallback_description="RSS summary",
            )

        mock_get.assert_called_once()
        response.raise_for_status.assert_called_once()
        self.assertEqual(metadata.title, "Publisher headline")
        self.assertEqual(metadata.description, "Publisher summary")
        self.assertEqual(metadata.image_url, "https://example.com/image.jpg")

    def test_fetch_link_card_metadata_falls_back_on_failure(self) -> None:
        with patch(
            "link_preview.requests.get",
            side_effect=requests.RequestException("metadata failed"),
        ):
            metadata = fetch_link_card_metadata(
                "https://example.com/story",
                fallback_title="RSS headline",
                fallback_description="RSS summary",
            )

        self.assertEqual(metadata.title, "RSS headline")
        self.assertEqual(metadata.description, "RSS summary")
        self.assertIsNone(metadata.image_url)

    def test_post_to_bluesky_with_news_url_sends_external_embed(self) -> None:
        tmp_dir, config = load_temp_config(
            POST_TO_BLUESKY="true",
            BLUESKY_HANDLE="example.bsky.social",
            BLUESKY_APP_PASSWORD="app-password",
        )
        self.addCleanup(tmp_dir.cleanup)
        client = MagicMock()
        client.upload_blob.return_value = SimpleNamespace(
            blob={"ref": {"$link": "blob-ref"}, "mimeType": "image/jpeg", "size": 10}
        )
        client.send_post.return_value = SimpleNamespace(
            uri="at://did:plc:abc123/app.bsky.feed.post/3k4duaz5vfs2b",
            cid="bafyexample",
        )
        metadata_response = MagicMock(
            url="https://example.com/story",
            text=(
                '<meta property="og:title" content="Publisher headline">'
                '<meta property="og:description" content="Publisher summary">'
                '<meta property="og:image" content="https://example.com/image.jpg">'
            ),
        )
        image_response = MagicMock(
            headers={"Content-Type": "image/jpeg"},
            content=b"image-bytes",
        )

        with patch("bluesky_publisher.Client", return_value=client):
            with patch(
                "link_preview.requests.get",
                side_effect=[metadata_response, image_response],
            ):
                post_to_bluesky(
                    config,
                    "Fresh take 🚀 #botWrites",
                    news_url="https://example.com/story",
                    news_title="RSS headline",
                    news_summary="RSS summary",
                )

        client.upload_blob.assert_called_once_with(b"image-bytes")
        call_kwargs = client.send_post.call_args.kwargs
        self.assertEqual(client.send_post.call_args.args[0], "Fresh take 🚀 #botWrites")
        embed = call_kwargs["embed"]
        self.assertEqual(embed.external.uri, "https://example.com/story")
        self.assertEqual(embed.external.title, "Publisher headline")
        self.assertEqual(embed.external.description, "Publisher summary")
        self.assertEqual(embed.external.thumb.ref.link, "blob-ref")

    def test_post_to_bluesky_falls_back_to_rss_metadata(self) -> None:
        tmp_dir, config = load_temp_config(
            POST_TO_BLUESKY="true",
            BLUESKY_HANDLE="example.bsky.social",
            BLUESKY_APP_PASSWORD="app-password",
        )
        self.addCleanup(tmp_dir.cleanup)
        client = MagicMock()
        client.send_post.return_value = SimpleNamespace(
            uri="at://did:plc:abc123/app.bsky.feed.post/3k4duaz5vfs2b",
            cid="bafyexample",
        )

        with patch("bluesky_publisher.Client", return_value=client):
            with patch(
                "link_preview.requests.get",
                side_effect=requests.RequestException("metadata failed"),
            ):
                post_to_bluesky(
                    config,
                    "Fresh take 🚀 #botWrites",
                    news_url="https://example.com/story",
                    news_title="RSS headline",
                    news_summary="RSS summary",
                )

        embed = client.send_post.call_args.kwargs["embed"]
        self.assertEqual(embed.external.uri, "https://example.com/story")
        self.assertEqual(embed.external.title, "RSS headline")
        self.assertEqual(embed.external.description, "RSS summary")
        self.assertIsNone(embed.external.thumb)

    def test_post_to_bluesky_continues_when_thumbnail_upload_fails(self) -> None:
        tmp_dir, config = load_temp_config(
            POST_TO_BLUESKY="true",
            BLUESKY_HANDLE="example.bsky.social",
            BLUESKY_APP_PASSWORD="app-password",
        )
        self.addCleanup(tmp_dir.cleanup)
        client = MagicMock()
        client.upload_blob.side_effect = RuntimeError("upload failed")
        client.send_post.return_value = SimpleNamespace(
            uri="at://did:plc:abc123/app.bsky.feed.post/3k4duaz5vfs2b",
            cid="bafyexample",
        )
        metadata_response = MagicMock(
            url="https://example.com/story",
            text='<meta property="og:image" content="https://example.com/image.jpg">',
        )
        image_response = MagicMock(
            headers={"Content-Type": "image/jpeg"},
            content=b"image-bytes",
        )

        with patch("bluesky_publisher.Client", return_value=client):
            with patch(
                "link_preview.requests.get",
                side_effect=[metadata_response, image_response],
            ):
                post_to_bluesky(
                    config,
                    "Fresh take 🚀 #botWrites",
                    news_url="https://example.com/story",
                    news_title="RSS headline",
                    news_summary="RSS summary",
                )

        embed = client.send_post.call_args.kwargs["embed"]
        self.assertIsNone(embed.external.thumb)

    def test_post_to_bluesky_raises_clear_error_on_failure(self) -> None:
        tmp_dir, config = load_temp_config(
            POST_TO_BLUESKY="true",
            BLUESKY_HANDLE="example.bsky.social",
            BLUESKY_APP_PASSWORD="app-password",
        )
        self.addCleanup(tmp_dir.cleanup)
        client = MagicMock()
        client.send_post.side_effect = RuntimeError("rate limited")

        with patch("bluesky_publisher.Client", return_value=client):
            with self.assertRaisesRegex(RuntimeError, "Bluesky posting failed: rate limited"):
                post_to_bluesky(config, "Fresh take 🚀 #botWrites")

    def test_post_to_bluesky_requires_post_uri(self) -> None:
        tmp_dir, config = load_temp_config(
            POST_TO_BLUESKY="true",
            BLUESKY_HANDLE="example.bsky.social",
            BLUESKY_APP_PASSWORD="app-password",
        )
        self.addCleanup(tmp_dir.cleanup)
        client = MagicMock()
        client.send_post.return_value = SimpleNamespace(uri=None, cid="bafyexample")

        with patch("bluesky_publisher.Client", return_value=client):
            with self.assertRaisesRegex(
                RuntimeError, "Bluesky API response did not include a post URI"
            ):
                post_to_bluesky(config, "Fresh take 🚀 #botWrites")
