from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from on_demand_requests import (
    DiscordMessageSnapshot,
    fetch_news_item_from_url,
    parse_on_demand_command,
    select_on_demand_request,
)
from support import load_temp_config


class OnDemandRequestTests(unittest.TestCase):
    def test_parse_post_single_line(self) -> None:
        tmp_dir, config = load_temp_config()
        self.addCleanup(tmp_dir.cleanup)

        request = parse_on_demand_command(
            "/post This is the exact post text.",
            config,
        )

        self.assertEqual(request.kind, "direct_post")
        self.assertEqual(request.post_text, "This is the exact post text.")

    def test_parse_post_multiline(self) -> None:
        tmp_dir, config = load_temp_config()
        self.addCleanup(tmp_dir.cleanup)

        request = parse_on_demand_command(
            "/post\nThis is the exact post text.\nKeep this line too.",
            config,
        )

        self.assertEqual(request.kind, "direct_post")
        self.assertEqual(
            request.post_text,
            "This is the exact post text.\nKeep this line too.",
        )

    def test_parse_news_url_with_tone(self) -> None:
        tmp_dir, config = load_temp_config(TONES="witty,analysis,deep thought")
        self.addCleanup(tmp_dir.cleanup)

        request = parse_on_demand_command(
            "/news https://example.com/story tone=deep thought",
            config,
        )

        self.assertEqual(request.kind, "news_url")
        self.assertEqual(request.news_url, "https://example.com/story")
        self.assertEqual(request.tone, "deep thought")

    def test_parse_news_url_with_invalid_tone_leaves_tone_unset(self) -> None:
        tmp_dir, config = load_temp_config(TONES="witty,analysis")
        self.addCleanup(tmp_dir.cleanup)

        request = parse_on_demand_command(
            "/news https://example.com/story tone=angry",
            config,
        )

        self.assertEqual(request.news_url, "https://example.com/story")
        self.assertIsNone(request.tone)

    def test_select_rejects_unauthorized_author(self) -> None:
        tmp_dir, config = load_temp_config(DISCORD_APPROVER_USER_IDS="111")
        self.addCleanup(tmp_dir.cleanup)

        selected = select_on_demand_request(
            [
                DiscordMessageSnapshot(
                    message_id="1",
                    author_id="999",
                    author_is_bot=False,
                    content="/post Unauthorized post.",
                )
            ],
            config,
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected.message_id, "1")
        self.assertIn("not allowed", selected.error or "")

    def test_select_skips_messages_already_replied_to_by_bot(self) -> None:
        tmp_dir, config = load_temp_config(DISCORD_APPROVER_USER_IDS="111")
        self.addCleanup(tmp_dir.cleanup)

        selected = select_on_demand_request(
            [
                DiscordMessageSnapshot(
                    message_id="1",
                    author_id="111",
                    author_is_bot=False,
                    content="/post Already handled.",
                ),
                DiscordMessageSnapshot(
                    message_id="2",
                    author_id="bot",
                    author_is_bot=True,
                    content="Picked up this on-demand request.",
                    referenced_message_id="1",
                ),
                DiscordMessageSnapshot(
                    message_id="3",
                    author_id="111",
                    author_is_bot=False,
                    content="/news https://example.com/story",
                ),
            ],
            config,
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected.message_id, "3")
        self.assertEqual(selected.request.kind, "news_url")

    def test_select_prioritizes_direct_post_before_news_url(self) -> None:
        tmp_dir, config = load_temp_config(DISCORD_APPROVER_USER_IDS="111")
        self.addCleanup(tmp_dir.cleanup)

        selected = select_on_demand_request(
            [
                DiscordMessageSnapshot(
                    message_id="1",
                    author_id="111",
                    author_is_bot=False,
                    content="/news https://example.com/story",
                ),
                DiscordMessageSnapshot(
                    message_id="2",
                    author_id="111",
                    author_is_bot=False,
                    content="/post Direct post wins.",
                ),
            ],
            config,
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected.message_id, "2")
        self.assertEqual(selected.request.kind, "direct_post")

    def test_fetch_news_item_from_url_reads_article_metadata(self) -> None:
        tmp_dir, config = load_temp_config()
        self.addCleanup(tmp_dir.cleanup)
        response = MagicMock(
            url="https://publisher.example.com/story",
            text=(
                '<meta property="og:title" content="Publisher headline">'
                '<meta property="og:description" content="Publisher summary">'
                '<meta property="og:site_name" content="Publisher">'
                '<meta property="article:published_time" '
                'content="2026-06-27T15:23:00Z">'
            ),
        )
        response.raise_for_status.return_value = None

        with patch(
            "on_demand_requests.resolve_news_url",
            return_value="https://publisher.example.com/story",
        ):
            with patch("on_demand_requests.requests.get", return_value=response):
                news_item = fetch_news_item_from_url(
                    "https://news.google.com/rss/articles/example",
                    config,
                    now=datetime(2026, 6, 28, 12, 0, tzinfo=timezone.utc),
                )

        self.assertEqual(news_item.title, "Publisher headline")
        self.assertEqual(news_item.source, "Publisher")
        self.assertEqual(news_item.summary, "Publisher summary")
        self.assertEqual(news_item.link, "https://publisher.example.com/story")
        self.assertEqual(
            news_item.published_at,
            datetime(2026, 6, 27, 15, 23, tzinfo=timezone.utc),
        )


if __name__ == "__main__":
    unittest.main()
