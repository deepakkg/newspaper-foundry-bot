from __future__ import annotations

from datetime import datetime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30), "IST")


def format_datetime_ist(value: datetime) -> str:
    return value.astimezone(IST).strftime("%Y-%m-%d %H:%M IST")
