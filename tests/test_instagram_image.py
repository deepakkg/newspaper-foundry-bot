from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import instagram_image
from instagram_image import (
    FOOTER_SAFE_TOP_Y,
    IMAGE_SIZE,
    build_instagram_image_body_text,
    extract_emojis,
    render_instagram_image,
)


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

    def test_footer_uses_monospace_font_loader(self) -> None:
        footer_font = instagram_image._load_monospace_font(24)

        self.assertIsNotNone(footer_font)

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
            self.assertNotIn(("#botWrites", False), drawn_text)

    def test_render_instagram_image_draws_footer_with_dedicated_helper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "post.png"
            with patch.object(instagram_image, "_load_emoji_font", return_value=None):
                with patch.object(
                    instagram_image,
                    "_draw_footer_label",
                    wraps=instagram_image._draw_footer_label,
                ) as mock_footer:
                    render_instagram_image("Cricket changed the game 🏏", output_path)

        mock_footer.assert_called_once()
        self.assertEqual(mock_footer.call_args.args[1], "#botWrites")

    def test_render_instagram_image_keeps_emoji_above_footer_zone(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "post.png"
            emoji_font = MagicMock()
            emoji_font.getbbox.return_value = (0, 0, 40, 40)
            emoji_positions: list[int] = []

            def capture_centered_text(*args, **kwargs):
                if args[1] == "💻🚀":
                    emoji_positions.append(kwargs["y"])

            with patch.object(instagram_image, "_load_emoji_font", return_value=emoji_font):
                with patch.object(instagram_image, "_emoji_text_renders_cleanly", return_value=True):
                    with patch.object(
                        instagram_image,
                        "_draw_centered_text",
                        side_effect=capture_centered_text,
                    ):
                        render_instagram_image(
                            "Forward Deployed Engineers are just devs who can't hide "
                            "from the customers. Blockchain Council released a 2026 "
                            "prep guide for FDE interviews. Good luck explaining a "
                            "smart contract bug while the client stares at you. 💻🚀 #botWrites",
                            output_path,
                        )

            self.assertTrue(emoji_positions)
            self.assertLess(emoji_positions[0] + 58, FOOTER_SAFE_TOP_Y)

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

    def test_render_instagram_image_handles_footer_overlap_regression_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "post.png"
            render_instagram_image(
                "Forward Deployed Engineers are just devs who can't hide from the "
                "customers. Blockchain Council released a 2026 prep guide for FDE "
                "interviews. Good luck explaining a smart contract bug while the "
                "client stares at you. 💻🚀 #botWrites",
                output_path,
            )

            from PIL import Image

            with Image.open(output_path) as image:
                self.assertEqual(image.size, (1080, 1080))
