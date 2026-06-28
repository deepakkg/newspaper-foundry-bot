from __future__ import annotations

import unittest

from publisher import build_post_text, build_post_text_without_url, max_generated_text_chars


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
