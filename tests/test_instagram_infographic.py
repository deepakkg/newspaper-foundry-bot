from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

from instagram_infographic import (
    INFOGRAPHIC_HEIGHT,
    INFOGRAPHIC_WIDTH,
    InfographicBlock,
    InfographicPlan,
    build_instagram_infographic_plan,
    clean_infographic_text,
    fallback_infographic_plan,
    render_instagram_infographic,
    select_infographic_style,
)
from news_fetcher import NewsItem
from support import chat_response, load_temp_config


def sample_news() -> NewsItem:
    return NewsItem(
        title="AI agents reshape support workflows",
        source="Example News",
        published_at=datetime(2026, 5, 31, 10, 0, tzinfo=timezone.utc),
        link="https://example.com/ai-agents",
        summary="Companies are deploying agents to resolve support tickets.",
    )


class InstagramInfographicTests(unittest.TestCase):
    def test_select_infographic_style_uses_direct_post_editorial(self) -> None:
        self.assertEqual(
            select_infographic_style(
                "auto",
                topic="on-demand post",
                tone="direct",
                post_text="Use this exact quote.",
                news_item=None,
                is_direct_post=True,
            ),
            "foundry_editorial",
        )

    def test_select_infographic_style_uses_schematic_for_technical_content(self) -> None:
        self.assertEqual(
            select_infographic_style(
                "auto",
                topic="saas professional services",
                tone="analysis",
                post_text="SaaS onboarding is churn prevention.",
                news_item=sample_news(),
                is_direct_post=False,
            ),
            "foundry_schematic",
        )

    def test_clean_infographic_text_removes_urls_hashtags_and_emojis(self) -> None:
        cleaned = clean_infographic_text(
            "AI agents need handoffs 🤖 #botWrites https://example.com/story"
        )

        self.assertEqual(cleaned, "AI agents need handoffs")

    def test_direct_post_plan_uses_supplied_text_without_llm_rewrite(self) -> None:
        tmp_dir, config = load_temp_config(INSTAGRAM_IMAGE_RENDERER="infographic")
        self.addCleanup(tmp_dir.cleanup)
        client = MagicMock()
        post_text = "This is the exact post text."

        plan = build_instagram_infographic_plan(
            client,
            config,
            topic="on-demand post",
            tone="direct",
            post_text=post_text,
            news_item=None,
            is_direct_post=True,
        )

        client.chat.completions.create.assert_not_called()
        self.assertEqual(plan.source_kind, "direct_post")
        self.assertEqual(plan.blocks[0].text, post_text)

    def test_news_backed_plan_uses_valid_llm_json(self) -> None:
        tmp_dir, config = load_temp_config(INSTAGRAM_IMAGE_RENDERER="infographic")
        self.addCleanup(tmp_dir.cleanup)
        client = MagicMock()
        client.chat.completions.create.return_value = chat_response(
            """
            {
              "title": "Support agents move into queues",
              "blocks": [
                {"label": "Trigger", "text": "Companies are deploying agents."},
                {"label": "Risk", "text": "Bad handoffs create support debt."}
              ],
              "takeaway": "Automation still needs ownership."
            }
            """
        )

        plan = build_instagram_infographic_plan(
            client,
            config,
            topic="ai agents",
            tone="analysis",
            post_text="AI agents need better handoffs.",
            news_item=sample_news(),
            is_direct_post=False,
        )

        self.assertEqual(plan.title, "Support agents move into queues")
        self.assertEqual(len(plan.blocks), 2)
        self.assertEqual(plan.takeaway, "Automation still needs ownership.")
        self.assertNotIn("https://", str(plan))
        self.assertNotIn("#botWrites", str(plan))

    def test_invalid_llm_plan_falls_back_safely(self) -> None:
        tmp_dir, config = load_temp_config(INSTAGRAM_IMAGE_RENDERER="infographic")
        self.addCleanup(tmp_dir.cleanup)
        client = MagicMock()
        client.chat.completions.create.return_value = chat_response("not json")

        plan = build_instagram_infographic_plan(
            client,
            config,
            topic="ai agents",
            tone="analysis",
            post_text="AI agents need better handoffs.",
            news_item=sample_news(),
            is_direct_post=False,
        )

        self.assertEqual(plan.source_kind, "news")
        self.assertGreaterEqual(len(plan.blocks), 2)
        self.assertNotIn("https://", str(plan))

    def test_render_instagram_infographic_creates_portrait_png_for_all_styles(self) -> None:
        for style in ("foundry_editorial", "foundry_schematic", "foundry_briefing"):
            with self.subTest(style=style):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    output_path = Path(tmp_dir) / f"{style}.png"
                    plan = InfographicPlan(
                        title="SaaS onboarding is churn prevention",
                        blocks=[
                            InfographicBlock("Signal", "Teams adopt products during onboarding."),
                            InfographicBlock("Risk", "Late handoffs create churn."),
                            InfographicBlock("Move", "Treat services as activation work."),
                        ],
                        takeaway="Revenue follows usage, not slide decks.",
                        style=style,
                        source_kind="news",
                    )

                    render_instagram_infographic(plan, output_path)

                    from PIL import Image

                    with Image.open(output_path) as image:
                        self.assertEqual(image.size, (INFOGRAPHIC_WIDTH, INFOGRAPHIC_HEIGHT))
                        self.assertEqual(image.format, "PNG")

    def test_fallback_plan_removes_article_url_and_hashtags(self) -> None:
        plan = fallback_infographic_plan(
            topic="ai agents",
            tone="analysis",
            post_text="AI agents need handoffs 🤖 #botWrites https://example.com/story",
            news_item=None,
            style="foundry_briefing",
            is_direct_post=False,
        )

        self.assertNotIn("https://", str(plan))
        self.assertNotIn("#botWrites", str(plan))
