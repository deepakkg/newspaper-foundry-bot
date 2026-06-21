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

        with patch("google_news_resolver.requests.get", return_value=response) as mock_get:
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

        with patch("google_news_resolver.requests.get", return_value=response):
            with patch(
                "google_news_resolver.requests.post",
                side_effect=requests.RequestException("decode failed"),
            ):
                resolved_url = resolve_news_url(google_url, timeout_seconds=120)

        self.assertEqual(resolved_url, google_url)

    def test_resolve_news_url_decodes_embedded_publisher_url(self) -> None:
        token = (
            base64.urlsafe_b64encode(b'\x08\x13"https://example.com/embedded-story"')
            .decode("ascii")
            .rstrip("=")
        )
        google_url = f"https://news.google.com/rss/articles/{token}"
        response = MagicMock(url=f"https://news.google.com/read/{token}")

        with patch("google_news_resolver.requests.get", return_value=response):
            with patch("google_news_resolver.requests.post") as mock_post:
                resolved_url = resolve_news_url(google_url, timeout_seconds=120)

        self.assertEqual(resolved_url, "https://example.com/embedded-story")
        mock_post.assert_not_called()

    def test_resolve_news_url_uses_google_decode_endpoint(self) -> None:
        token = "CBMiNewStyleToken"
        google_url = f"https://news.google.com/rss/articles/{token}"
        redirect_response = MagicMock(url=f"https://news.google.com/read/{token}")
        decode_response = MagicMock(
            text=')]}\n["wrb.fr","Fbv4je","[[\\"https:\\\\/\\\\/example.com\\\\/decoded-story\\"]]"]'
        )

        with patch("google_news_resolver.requests.get", return_value=redirect_response):
            with patch(
                "google_news_resolver.requests.post",
                return_value=decode_response,
            ) as mock_post:
                resolved_url = resolve_news_url(google_url, timeout_seconds=120)

        self.assertEqual(resolved_url, "https://example.com/decoded-story")
        mock_post.assert_called_once()
        post_kwargs = mock_post.call_args.kwargs
        self.assertEqual(post_kwargs["timeout"], 10)
        self.assertIn("f.req", post_kwargs["data"])
        redirect_response.raise_for_status.assert_called_once()
        decode_response.raise_for_status.assert_called_once()

    def test_resolve_news_url_uses_signed_google_decode_endpoint(self) -> None:
        token = "CBMiSignedStyleToken"
        google_url = f"https://news.google.com/rss/articles/{token}"
        redirect_response = MagicMock(
            url=f"https://news.google.com/read/{token}",
            text=(
                '<div data-n-a-id="CBMiSignedStyleToken" '
                'data-n-a-ts="1781602109" '
                'data-n-a-sg="AVvZt1GQ3mXICXiYhcZoLDoL59Hz"></div>'
            ),
        )
        decode_response = MagicMock(
            text=(
                ')]}\'\n\n[["wrb.fr","Fbv4je",'
                '"[\\"garturlres\\",\\"https://example.com/signed-story\\",1]",'
                'null,null,null,""]]'
            )
        )

        with patch("google_news_resolver.requests.get", return_value=redirect_response):
            with patch(
                "google_news_resolver.requests.post",
                return_value=decode_response,
            ) as mock_post:
                resolved_url = resolve_news_url(google_url, timeout_seconds=120)

        self.assertEqual(resolved_url, "https://example.com/signed-story")
        post_kwargs = mock_post.call_args.kwargs
        self.assertIsInstance(post_kwargs["data"], str)
        decoded_body = unquote(post_kwargs["data"])
        self.assertIn("garturlreq", decoded_body)
        self.assertIn(token, decoded_body)
        self.assertIn("1781602109", decoded_body)
        self.assertIn("AVvZt1GQ3mXICXiYhcZoLDoL59Hz", decoded_body)
        redirect_response.raise_for_status.assert_called_once()
        decode_response.raise_for_status.assert_called_once()

    def test_resolve_news_url_uses_decoded_google_article_id(self) -> None:
        article_id = "AU_yqLMexampleArticleId"
        token = (
            base64.urlsafe_b64encode(f'\x08\x13"{article_id}'.encode("utf-8"))
            .decode("ascii")
            .rstrip("=")
        )
        google_url = f"https://news.google.com/rss/articles/{token}"
        redirect_response = MagicMock(url=f"https://news.google.com/read/{token}")
        decode_response = MagicMock(text='["https:\\/\\/example.com\\/decoded-id-story"]')

        with patch("google_news_resolver.requests.get", return_value=redirect_response):
            with patch(
                "google_news_resolver.requests.post",
                return_value=decode_response,
            ) as mock_post:
                resolved_url = resolve_news_url(google_url, timeout_seconds=120)

        self.assertEqual(resolved_url, "https://example.com/decoded-id-story")
        payload = mock_post.call_args.kwargs["data"]["f.req"]
        self.assertIn(article_id, payload)
        self.assertNotIn(token, payload)

    def test_resolve_news_url_keeps_google_url_when_decode_fails(self) -> None:
        token = "CBMiNewStyleToken"
        google_url = f"https://news.google.com/rss/articles/{token}"
        response = MagicMock(url=f"https://news.google.com/read/{token}")

        with patch("google_news_resolver.requests.get", return_value=response):
            with patch(
                "google_news_resolver.requests.post",
                side_effect=requests.RequestException("decode failed"),
            ):
                resolved_url = resolve_news_url(google_url, timeout_seconds=120)

        self.assertEqual(resolved_url, google_url)

    def test_resolve_news_url_keeps_google_url_when_request_fails(self) -> None:
        google_url = "https://news.google.com/rss/articles/example"

        with patch(
            "google_news_resolver.requests.get",
            side_effect=requests.RequestException("network failed"),
        ):
            resolved_url = resolve_news_url(google_url, timeout_seconds=120)

        self.assertEqual(resolved_url, google_url)

    def test_resolve_news_url_skips_non_google_url(self) -> None:
        article_url = "https://example.com/story"

        with patch("google_news_resolver.requests.get") as mock_get:
            with patch("google_news_resolver.requests.post") as mock_post:
                resolved_url = resolve_news_url(article_url, timeout_seconds=120)

        self.assertEqual(resolved_url, article_url)
        mock_get.assert_not_called()
        mock_post.assert_not_called()

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
        rss_response = MagicMock()
        rss_response.text = rss
        with patch("news_fetcher.requests.get", return_value=rss_response) as mock_rss_get:
            with patch(
                "news_fetcher.resolve_news_url",
                return_value="https://example.com/newer",
            ) as mock_resolve:
                with patch("news_fetcher.datetime") as mock_datetime:
                    mock_datetime.now.return_value = datetime(
                        2026, 5, 31, 12, 0, tzinfo=timezone.utc
                    )
                    news_item = fetch_latest_news("learning", config)

        self.assertIsNotNone(news_item)
        self.assertEqual(news_item.title, "Newer learning headline")
        self.assertEqual(news_item.link, "https://example.com/newer")
        first_request_url = mock_rss_get.call_args.args[0]
        self.assertIn("q=learning", first_request_url)
        self.assertIn("hl=en-US", first_request_url)
        mock_resolve.assert_called_once_with(
            "https://news.google.com/rss/articles/newer",
            timeout_seconds=config.timeout_seconds,
        )
        rss_response.raise_for_status.assert_called_once()
