from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import publishing_flow
from config import load_config
from run_state import RunStateStore
from support import write_env_file


class PublishingStateTests(unittest.TestCase):
    def test_second_publish_for_same_content_skips_platform_api(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            write_env_file(
                env_path,
                POST_TO_BLUESKY="true",
                BLUESKY_HANDLE="example.bsky.social",
                BLUESKY_APP_PASSWORD="app-password",
            )
            config = load_config(env_path)
            store = RunStateStore(config.state_file_path)
            store.record_run("run-1", "overall", "ai agents", "witty")
            published = SimpleNamespace(
                uri="at://did/post/abc",
                url="https://bsky.app/profile/example.bsky.social/post/abc",
            )
            with patch.object(
                publishing_flow, "post_to_bluesky", return_value=published
            ) as mock_publish:
                first = publishing_flow.publish_enabled_platforms(
                    config,
                    topic="ai agents",
                    tweet="AI agents need better handoffs. 🤖",
                    final_post_text="AI agents need better handoffs. 🤖 #botWrites",
                    news_item=None,
                    instagram_caption=None,
                    run_id="run-1",
                    state_store=store,
                )
                second = publishing_flow.publish_enabled_platforms(
                    config,
                    topic="ai agents",
                    tweet="AI agents need better handoffs. 🤖",
                    final_post_text="AI agents need better handoffs. 🤖 #botWrites",
                    news_item=None,
                    instagram_caption=None,
                    run_id="run-2",
                    state_store=store,
                )

        self.assertEqual(first.results[0].status, "published")
        self.assertEqual(second.results[0].status, "already published")
        mock_publish.assert_called_once()


if __name__ == "__main__":
    unittest.main()
