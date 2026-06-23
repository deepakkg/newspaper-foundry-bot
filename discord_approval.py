from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal

from config import AppConfig
from news_fetcher import NewsItem
from notifications import format_news_published_at


ApprovalStatus = Literal["approved", "declined", "expired"]


@dataclass(frozen=True)
class ApprovalRequest:
    topic: str
    tone: str
    final_post_text: str
    instagram_caption: str | None
    elapsed: float
    attempts: int
    target_platforms: list[str]
    news_item: NewsItem | None = None


@dataclass(frozen=True)
class ApprovalDecision:
    status: ApprovalStatus
    user_id: str | None = None
    username: str | None = None


def is_authorized_approver(config: AppConfig, user_id: int | str) -> bool:
    return str(user_id) in set(config.discord_approver_user_ids)


def _field_value(value: str | None) -> str:
    return value if value else "Not available"


def build_approval_embed(request: ApprovalRequest) -> dict[str, object]:
    fields: list[dict[str, object]] = [
        {"name": "Topic", "value": request.topic, "inline": True},
        {"name": "Tone", "value": request.tone, "inline": True},
        {"name": "Attempts", "value": str(request.attempts), "inline": True},
        {
            "name": "Time taken",
            "value": f"{request.elapsed:.2f} seconds",
            "inline": True,
        },
        {
            "name": "Target platforms",
            "value": ", ".join(request.target_platforms) or "None",
            "inline": False,
        },
    ]
    if request.news_item:
        fields.extend(
            [
                {
                    "name": "News title",
                    "value": _field_value(request.news_item.title),
                    "inline": False,
                },
                {
                    "name": "News source",
                    "value": _field_value(request.news_item.source),
                    "inline": True,
                },
                {
                    "name": "News published",
                    "value": format_news_published_at(request.news_item),
                    "inline": True,
                },
                {
                    "name": "Article URL",
                    "value": _field_value(request.news_item.link),
                    "inline": False,
                },
            ]
        )
    fields.append(
        {
            "name": "Final post",
            "value": request.final_post_text[:1024],
            "inline": False,
        }
    )
    if request.instagram_caption:
        fields.append(
            {
                "name": "Instagram caption preview",
                "value": request.instagram_caption[:1024],
                "inline": False,
            }
        )
    return {"title": "Post awaiting approval", "color": 0xF1C40F, "fields": fields}


def parse_discord_channel_id(value: str | None) -> int:
    try:
        channel_id = int(value or "")
    except ValueError as exc:
        raise RuntimeError("DISCORD_CHANNEL_ID must be a numeric Discord channel ID.") from exc
    if channel_id <= 0:
        raise RuntimeError("DISCORD_CHANNEL_ID must be a numeric Discord channel ID.")
    return channel_id


def request_discord_approval(
    config: AppConfig,
    approval_request: ApprovalRequest,
) -> ApprovalDecision:
    return asyncio.run(_request_discord_approval(config, approval_request))


async def _request_discord_approval(
    config: AppConfig,
    approval_request: ApprovalRequest,
) -> ApprovalDecision:
    try:
        import discord
    except ImportError as exc:
        raise RuntimeError("discord.py package is not installed.") from exc

    channel_id = parse_discord_channel_id(config.discord_channel_id)
    intents = discord.Intents.default()
    client = discord.Client(intents=intents)
    decision: ApprovalDecision = ApprovalDecision(status="expired")
    ready_event = asyncio.Event()
    decision_event = asyncio.Event()
    sent_message = None
    ready_error: Exception | None = None

    class ApprovalView(discord.ui.View):
        def __init__(self) -> None:
            super().__init__(timeout=config.approval_timeout_minutes * 60)

        async def _finish(
            self,
            interaction,
            status: ApprovalStatus,
            confirmation: str,
        ) -> None:
            nonlocal decision
            if not is_authorized_approver(config, interaction.user.id):
                await interaction.response.send_message(
                    "You are not allowed to approve or decline this post.",
                    ephemeral=True,
                )
                return

            for item in self.children:
                item.disabled = True
            decision = ApprovalDecision(
                status=status,
                user_id=str(interaction.user.id),
                username=str(interaction.user),
            )
            await interaction.response.edit_message(view=self)
            await interaction.followup.send(confirmation)
            decision_event.set()

        @discord.ui.button(label="Approve", style=discord.ButtonStyle.success)
        async def approve(self, interaction, _button) -> None:
            await self._finish(interaction, "approved", "Post approved. Publishing now.")

        @discord.ui.button(label="Decline", style=discord.ButtonStyle.danger)
        async def decline(self, interaction, _button) -> None:
            await self._finish(interaction, "declined", "Post was not published.")

    view = ApprovalView()

    @client.event
    async def on_ready() -> None:
        nonlocal ready_error, sent_message
        try:
            channel = client.get_channel(channel_id)
            if channel is None:
                channel = await client.fetch_channel(channel_id)
            embed = discord.Embed.from_dict(build_approval_embed(approval_request))
            sent_message = await channel.send(embed=embed, view=view)
        except Exception as exc:
            ready_error = exc
        finally:
            ready_event.set()

    client_task = asyncio.create_task(client.start(config.discord_bot_token or ""))
    ready_task = asyncio.create_task(ready_event.wait())
    try:
        done, _pending = await asyncio.wait(
            {client_task, ready_task},
            timeout=60,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if client_task in done:
            client_task.result()
        if ready_task not in done:
            raise RuntimeError("Discord approval bot did not become ready.")
        if ready_error is not None:
            raise RuntimeError(
                f"Discord approval setup failed: {ready_error}"
            ) from ready_error
        await asyncio.wait_for(
            decision_event.wait(),
            timeout=config.approval_timeout_minutes * 60,
        )
    except asyncio.TimeoutError:
        for item in view.children:
            item.disabled = True
        if sent_message is not None:
            await sent_message.edit(view=view)
            await sent_message.channel.send("Approval expired. Post was not published.")
    finally:
        ready_task.cancel()
        await client.close()
        try:
            await client_task
        except asyncio.CancelledError:
            pass
    return decision
