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
