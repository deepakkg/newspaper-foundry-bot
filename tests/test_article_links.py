from __future__ import annotations

import unittest
from datetime import datetime, timezone

from article_links import build_article_link_entry, update_article_links_page
from news_fetcher import NewsItem
from support import load_temp_config


class ArticleLinksTests(unittest.TestCase):
    def test_update_article_links_page_adds_resolved_publisher_url(self) -> None:
        tmp_dir, config = load_temp_config(
            ARTICLE_LINKS_ENABLED="true",
            ARTICLE_LINKS_MAX_ITEMS="25",
        )
        self.addCleanup(tmp_dir.cleanup)
        news_item = NewsItem(
            title="AI agents reshape support workflows",
            source="Example News",
            published_at=datetime(2026, 5, 31, 10, 0, tzinfo=timezone.utc),
            link="https://publisher.example.com/ai-agents",
            summary="Companies are deploying agents to resolve support tickets.",
        )

        update_article_links_page(
            config,
            build_article_link_entry(
                news_item,
                instagram_media_id="179",
                instagram_url="https://instagram.com/p/abc",
            ),
        )

        json_payload = config.article_links_data_path.read_text(encoding="utf-8")
        html = config.article_links_html_path.read_text(encoding="utf-8")
        self.assertIn("https://publisher.example.com/ai-agents", json_payload)
        self.assertIn("https://publisher.example.com/ai-agents", html)
        self.assertNotIn("news.google.com/rss", json_payload)
        self.assertIn("AI agents reshape support workflows", html)
        self.assertIn("Published At: 2026-05-31 15:30 IST", html)

    def test_update_article_links_page_deduplicates_by_url(self) -> None:
        tmp_dir, config = load_temp_config(ARTICLE_LINKS_ENABLED="true")
        self.addCleanup(tmp_dir.cleanup)
        first_news = NewsItem(
            title="Original title",
            source="Example News",
            published_at=datetime(2026, 5, 31, 10, 0, tzinfo=timezone.utc),
            link="https://publisher.example.com/story",
            summary="Original summary.",
        )
        second_news = NewsItem(
            title="Updated title",
            source="Example News",
            published_at=datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc),
            link="https://publisher.example.com/story",
            summary="Updated summary.",
        )

        update_article_links_page(config, build_article_link_entry(first_news))
        update_article_links_page(config, build_article_link_entry(second_news))

        payload = config.article_links_data_path.read_text(encoding="utf-8")
        self.assertEqual(payload.count("https://publisher.example.com/story"), 1)
        self.assertIn("Updated title", payload)
        self.assertNotIn("Original title", payload)

    def test_update_article_links_page_keeps_latest_max_items(self) -> None:
        tmp_dir, config = load_temp_config(
            ARTICLE_LINKS_ENABLED="true",
            ARTICLE_LINKS_MAX_ITEMS="2",
        )
        self.addCleanup(tmp_dir.cleanup)

        for index in range(3):
            news_item = NewsItem(
                title=f"Story {index}",
                source="Example News",
                published_at=datetime(2026, 5, 31, 10, index, tzinfo=timezone.utc),
                link=f"https://publisher.example.com/story-{index}",
                summary="Summary.",
            )
            update_article_links_page(config, build_article_link_entry(news_item))

        payload = config.article_links_data_path.read_text(encoding="utf-8")
        self.assertIn("story-2", payload)
        self.assertIn("story-1", payload)
        self.assertNotIn("story-0", payload)
