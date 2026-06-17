from __future__ import annotations

from typing import Any

import requests

from config import AppConfig


def send_discord_embed(config: AppConfig, embed: dict[str, Any]) -> None:
    if not config.discord_webhook_url:
        return

    response = requests.post(
        config.discord_webhook_url,
        json={
            "embeds": [embed],
            "allowed_mentions": {"parse": []},
        },
        timeout=config.timeout_seconds,
    )
    if response.status_code >= 400:
        detail = response.text.strip() or response.reason
        raise RuntimeError(f"Discord send failed: {detail}")
