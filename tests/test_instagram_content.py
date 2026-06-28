from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from instagram_content import (
    build_instagram_caption,
    extract_hashtags,
    fallback_hashtags,
    format_caption_news_title,
    generate_instagram_hashtags,
)
from news_fetcher import NewsItem
from support import chat_response, load_temp_config


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

        lines = caption.splitlines()
        self.assertEqual(lines[0], "AI agents reshape support workflows")
        self.assertIn("Source: Example News", caption)
        self.assertIn("Published At: 2026-05-31 15:30 IST", caption)
        self.assertIn("#aiagents #analysis #AI #SupportOps #botWrites", caption)
        self.assertNotIn("News title:", caption)
        self.assertNotIn("News source:", caption)
        self.assertNotIn("News published:", caption)
        self.assertNotIn("Hashtags:", caption)
        self.assertNotIn("Topic:", caption)
        self.assertNotIn("Tone:", caption)
        self.assertNotIn("https://example.com/ai-agents", caption)
        self.assertTrue(caption.strip().endswith("#botWrites"))

    def test_build_instagram_caption_adds_article_link_in_bio_for_news_when_enabled(
        self,
    ) -> None:
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
            llm_hashtags=["#AI"],
            article_link_in_bio=True,
        )

        lines = caption.splitlines()
        self.assertEqual(lines[2], "Published At: 2026-05-31 15:30 IST")
        self.assertEqual(lines[3], "Article link in bio.")
        self.assertEqual(lines[4], "")
        self.assertEqual(lines[5], "#aiagents #analysis #AI #botWrites")
        self.assertTrue(caption.strip().endswith("#aiagents #analysis #AI #botWrites"))
        self.assertIn("#aiagents #analysis #AI #botWrites", caption)
        self.assertNotIn("https://example.com/ai-agents", caption)

    def test_build_instagram_caption_uses_hashtags_only_without_news(self) -> None:
        caption = build_instagram_caption(
            topic="saas professional services",
            tone="analysis",
            news_item=None,
            llm_hashtags=["#SaaS", "#BusinessAnalysis"],
            article_link_in_bio=True,
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
