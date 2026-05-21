from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from config import AppConfig
from logger import has_logged_slot


FIXED_RUN_SLOTS = ("06:00", "12:00", "18:00", "22:00")
DEFAULT_SLOT_WINDOW_MINUTES = 90


@dataclass(frozen=True)
class RunDecision:
    should_run: bool
    run_date: str
    run_slot: str
    reason: str


def _slot_datetime(now: datetime, slot: str) -> datetime:
    hour, minute = (int(part) for part in slot.split(":", maxsplit=1))
    return now.replace(hour=hour, minute=minute, second=0, microsecond=0)


def resolve_current_slot(
    now: datetime,
    fixed_slots: tuple[str, ...] = FIXED_RUN_SLOTS,
    window_minutes: int = DEFAULT_SLOT_WINDOW_MINUTES,
) -> tuple[str, str] | None:
    candidates: list[tuple[timedelta, datetime, str]] = []
    for slot in fixed_slots:
        slot_time = _slot_datetime(now, slot)
        if slot_time > now:
            slot_time -= timedelta(days=1)
        age = now - slot_time
        if timedelta(0) <= age <= timedelta(minutes=window_minutes):
            candidates.append((age, slot_time, slot))

    if not candidates:
        return None

    _, slot_time, slot = min(candidates, key=lambda item: item[0])
    return slot_time.date().isoformat(), slot


def decide_scheduled_run(config: AppConfig, now: datetime | None = None) -> RunDecision:
    timezone = ZoneInfo(config.run_timezone)
    current_time = now.astimezone(timezone) if now else datetime.now(timezone)
    resolved_slot = resolve_current_slot(current_time)

    if resolved_slot is None:
        return RunDecision(
            should_run=False,
            run_date=current_time.date().isoformat(),
            run_slot="unknown",
            reason="No fixed run slot is currently due.",
        )

    run_date, run_slot = resolved_slot
    if run_slot not in config.enabled_run_slots:
        return RunDecision(
            should_run=False,
            run_date=run_date,
            run_slot=run_slot,
            reason=f"Run slot {run_slot} is not enabled.",
        )

    if has_logged_slot(config.log_file_path, run_date=run_date, run_slot=run_slot):
        return RunDecision(
            should_run=False,
            run_date=run_date,
            run_slot=run_slot,
            reason=f"Run slot {run_date} {run_slot} is already logged.",
        )

    return RunDecision(
        should_run=True,
        run_date=run_date,
        run_slot=run_slot,
        reason=f"Run slot {run_date} {run_slot} is due.",
    )
