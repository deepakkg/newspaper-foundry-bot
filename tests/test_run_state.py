from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from run_state import RunStateStore, content_hash, new_run_id


class RunStateStoreTests(unittest.TestCase):
    def test_new_run_id_is_unique_and_nonempty(self) -> None:
        first = new_run_id()
        second = new_run_id()

        self.assertTrue(first)
        self.assertNotEqual(first, second)

    def test_content_hash_is_deterministic_and_changes_with_payload(self) -> None:
        first = content_hash("Bluesky", "same post", "https://example.com/a")
        second = content_hash("Bluesky", "same post", "https://example.com/a")
        changed = content_hash("Bluesky", "changed post", "https://example.com/a")

        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)

    def test_published_platform_can_be_found_across_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = RunStateStore(Path(tmp_dir) / "run-state.json")
            store.record_run("run-1", content_hash("X", "post", ""), "topic", "witty")
            fingerprint = content_hash("X", "post", "")
            store.record_platform_success(
                "run-1",
                platform="X",
                fingerprint=fingerprint,
                identifier="123",
                url="https://x.com/example/status/123",
            )

            found = store.find_published_platform(fingerprint, "X")

        self.assertEqual(found["identifier"], "123")
        self.assertEqual(found["url"], "https://x.com/example/status/123")

    def test_publishing_intent_is_visible_as_uncertain_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = RunStateStore(Path(tmp_dir) / "run-state.json")
            store.record_run("run-1", "hash-1", "topic", "tone")
            store.record_platform_intent(
                "run-1", platform="Bluesky", fingerprint="fingerprint-1"
            )

            state = store.find_platform_state("fingerprint-1", "Bluesky")
            self.assertIsNotNone(state)
            self.assertEqual(state["status"], "publishing")
            self.assertIsNone(
                store.find_published_platform("fingerprint-1", "Bluesky")
            )

    def test_claim_platform_is_atomic_across_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = RunStateStore(Path(tmp_dir) / "run-state.json")
            store.record_run("run-1", "hash-1", "topic", "tone")
            store.record_run("run-2", "hash-2", "topic", "tone")

            first_claim = store.claim_platform(
                "run-1", platform="X", fingerprint="fingerprint-1"
            )
            second_claim = store.claim_platform(
                "run-2", platform="X", fingerprint="fingerprint-1"
            )

        self.assertIsNone(first_claim)
        self.assertIsNotNone(second_claim)
        self.assertEqual(second_claim["run_id"], "run-1")
        self.assertEqual(second_claim["status"], "publishing")

    def test_state_writes_are_readable_after_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "nested" / "run-state.json"
            store = RunStateStore(path)
            store.record_run("run-1", "hash-1", "topic", "tone", request_id="discord-1")
            store.mark_request_consumed("discord-1", "run-1")

            reopened = RunStateStore(path)
            self.assertIn("discord-1", reopened.consumed_request_ids())
            self.assertEqual(reopened.get_run("run-1")["request_id"], "discord-1")


if __name__ == "__main__":
    unittest.main()
