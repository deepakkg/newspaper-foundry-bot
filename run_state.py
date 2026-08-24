from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator


STATE_VERSION = 1


def new_run_id() -> str:
    return uuid.uuid4().hex


def content_hash(platform: str, *parts: str | None) -> str:
    payload = {"platform": platform, "parts": [part or "" for part in parts]}
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


class RunStateStore:
    def __init__(self, path: Path):
        self.path = path
        self.lock_path = path.with_name(f".{path.name}.lock")

    def _empty_payload(self) -> dict[str, Any]:
        return {"version": STATE_VERSION, "runs": {}, "consumed_requests": {}}

    def _read_unlocked(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._empty_payload()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Could not read run state file {self.path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"Run state file {self.path} must contain a JSON object.")
        version = payload.get("version", STATE_VERSION)
        if version != STATE_VERSION:
            raise RuntimeError(
                f"Unsupported run state version {version}; expected {STATE_VERSION}."
            )
        runs = payload.get("runs", {})
        consumed_requests = payload.get("consumed_requests", {})
        if not isinstance(runs, dict) or not isinstance(consumed_requests, dict):
            raise RuntimeError(f"Run state file {self.path} has an invalid structure.")
        return {
            "version": version,
            "runs": runs,
            "consumed_requests": consumed_requests,
        }

    def _write_unlocked(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.path)
        except Exception:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise

    @contextmanager
    def _locked(self, *, write: bool) -> Iterator[dict[str, Any]]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX if write else fcntl.LOCK_SH)
            payload = self._read_unlocked()
            try:
                yield payload
                if write:
                    self._write_unlocked(payload)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _mutate(self, callback: Callable[[dict[str, Any]], None]) -> None:
        with self._locked(write=True) as payload:
            callback(payload)

    def _read(self) -> dict[str, Any]:
        with self._locked(write=False) as payload:
            return payload

    def record_run(
        self,
        run_id: str,
        overall_fingerprint: str,
        topic: str,
        tone: str,
        *,
        request_id: str | None = None,
    ) -> None:
        def mutate(payload: dict[str, Any]) -> None:
            run = payload["runs"].setdefault(run_id, {})
            run.update(
                {
                    "run_id": run_id,
                    "overall_fingerprint": overall_fingerprint,
                    "topic": topic,
                    "tone": tone,
                    "request_id": request_id,
                    "created_at": run.get("created_at") or _utc_timestamp(),
                    "updated_at": _utc_timestamp(),
                    "platforms": run.get("platforms", {}),
                }
            )

        self._mutate(mutate)

    def get_run(self, run_id: str) -> dict[str, Any]:
        return dict(self._read()["runs"].get(run_id, {}))

    def record_platform_intent(
        self, run_id: str, *, platform: str, fingerprint: str
    ) -> None:
        def mutate(payload: dict[str, Any]) -> None:
            run = payload["runs"].get(run_id)
            if run is None:
                raise RuntimeError(f"Cannot record platform intent for unknown run: {run_id}")
            existing = (run.setdefault("platforms", {})).get(platform)
            if existing and existing.get("status") == "published":
                return
            run["platforms"][platform] = {
                "status": "publishing",
                "fingerprint": fingerprint,
                "started_at": _utc_timestamp(),
            }
            run["updated_at"] = _utc_timestamp()

        self._mutate(mutate)

    def claim_platform(
        self, run_id: str, *, platform: str, fingerprint: str
    ) -> dict[str, Any] | None:
        """Atomically find an existing attempt or claim this fingerprint."""
        result: dict[str, Any] | None = None

        def mutate(payload: dict[str, Any]) -> None:
            nonlocal result
            for existing_run_id, existing_run in reversed(
                list(payload["runs"].items())
            ):
                existing = (existing_run.get("platforms") or {}).get(platform)
                if isinstance(existing, dict) and existing.get("fingerprint") == fingerprint:
                    result = {"run_id": existing_run_id, **existing}
                    return
            run = payload["runs"].get(run_id)
            if run is None:
                raise RuntimeError(f"Cannot claim platform for unknown run: {run_id}")
            run.setdefault("platforms", {})[platform] = {
                "status": "publishing",
                "fingerprint": fingerprint,
                "started_at": _utc_timestamp(),
            }
            run["updated_at"] = _utc_timestamp()

        self._mutate(mutate)
        return result

    def record_platform_success(
        self,
        run_id: str,
        *,
        platform: str,
        fingerprint: str,
        identifier: str | None,
        url: str | None,
    ) -> None:
        def mutate(payload: dict[str, Any]) -> None:
            run = payload["runs"].get(run_id)
            if run is None:
                raise RuntimeError(f"Cannot record platform result for unknown run: {run_id}")
            run.setdefault("platforms", {})[platform] = {
                "status": "published",
                "fingerprint": fingerprint,
                "identifier": identifier,
                "url": url,
                "published_at": _utc_timestamp(),
            }
            run["updated_at"] = _utc_timestamp()

        self._mutate(mutate)

    def record_platform_failure(
        self, run_id: str, *, platform: str, fingerprint: str, error: str
    ) -> None:
        def mutate(payload: dict[str, Any]) -> None:
            run = payload["runs"].get(run_id)
            if run is None:
                raise RuntimeError(f"Cannot record platform failure for unknown run: {run_id}")
            run.setdefault("platforms", {})[platform] = {
                "status": "failed",
                "fingerprint": fingerprint,
                "error": error,
                "failed_at": _utc_timestamp(),
            }
            run["updated_at"] = _utc_timestamp()

        self._mutate(mutate)

    def find_platform_state(self, fingerprint: str, platform: str) -> dict[str, Any] | None:
        payload = self._read()
        for run_id, run in reversed(list(payload["runs"].items())):
            result = (run.get("platforms") or {}).get(platform)
            if isinstance(result, dict) and result.get("fingerprint") == fingerprint:
                return {"run_id": run_id, **result}
        return None

    def find_published_platform(self, fingerprint: str, platform: str) -> dict[str, Any] | None:
        result = self.find_platform_state(fingerprint, platform)
        if result and result.get("status") == "published":
            return result
        return None

    def consumed_request_ids(self) -> set[str]:
        return set(self._read()["consumed_requests"])

    def mark_request_consumed(self, request_id: str, run_id: str) -> None:
        def mutate(payload: dict[str, Any]) -> None:
            payload["consumed_requests"][request_id] = {
                "run_id": run_id,
                "consumed_at": _utc_timestamp(),
            }

        self._mutate(mutate)
