from __future__ import annotations

import requests

from config import AppConfig


TELEGRAM_SEND_MESSAGE_URL = "https://api.telegram.org/bot{token}/sendMessage"


def send_telegram_message(config: AppConfig, message_text: str) -> None:
    if not config.telegram_bot_token or not config.telegram_chat_id:
        return

    response = requests.post(
        TELEGRAM_SEND_MESSAGE_URL.format(token=config.telegram_bot_token),
        json={
            "chat_id": config.telegram_chat_id,
            "text": message_text,
        },
        timeout=config.timeout_seconds,
    )
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError("Telegram API returned a non-JSON response.") from exc

    if response.status_code >= 400 or not payload.get("ok", False):
        detail = payload.get("description") or str(payload)
        raise RuntimeError(f"Telegram send failed: {detail}")
