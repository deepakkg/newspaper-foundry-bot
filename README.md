# Gemma Tweet Bot

This bot fetches recent topic news from Google News RSS, generates short posts with an OpenAI-compatible LLM API, optionally asks for Discord approval, then publishes to the enabled platforms: Bluesky, Instagram, X, or any combination of them.

## GitHub Actions schedule

The workflow has four explicit GitHub cron entries for these India-time slots:

- 08:00 IST
- 12:00 IST
- 16:00 IST
- 20:00 IST

GitHub cron uses UTC, so the workflow file maps these to `02:30`, `06:30`, `10:30`, and `14:30` UTC.

## GitHub Secrets

Store these in repository Settings -> Secrets and variables -> Actions -> Secrets:

- `LLM_API_KEY`
- `BLUESKY_APP_PASSWORD`
- `INSTAGRAM_ACCESS_TOKEN`
- `CLOUDINARY_API_KEY`
- `CLOUDINARY_API_SECRET`
- `X_API_KEY`
- `X_API_KEY_SECRET`
- `X_ACCESS_TOKEN`
- `X_ACCESS_TOKEN_SECRET`
- `DISCORD_BOT_TOKEN`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`
- `DISCORD_WEBHOOK_URL`

## GitHub Variables

Store these in repository Settings -> Secrets and variables -> Actions -> Variables:

- `LLM_BASE_URL`
- `LLM_MODEL`
- `TOPICS`
- `TONES`
- `MAX_TWEET_CHARS`
- `MAX_RETRIES`
- `LLM_TIMEOUT_SECONDS`
- `NEWS_ENABLED`
- `NEWS_RECENCY_HOURS`
- `NEWS_REGION`
- `NEWS_LANGUAGE`
- `POST_TO_BLUESKY`
- `BLUESKY_HANDLE`
- `BLUESKY_SERVICE_URL`
- `POST_TO_INSTAGRAM`
- `INSTAGRAM_ACCOUNT_ID`
- `INSTAGRAM_GRAPH_BASE_URL`
- `INSTAGRAM_GRAPH_API_VERSION`
- `CLOUDINARY_CLOUD_NAME`
- `CLOUDINARY_FOLDER`
- `POST_TO_X`
- `X_USERNAME`
- `APPROVAL_REQUIRED`
- `APPROVAL_TIMEOUT_MINUTES`
- `DISCORD_CHANNEL_ID`
- `DISCORD_APPROVER_USER_IDS`
- `TELEGRAM_NOTIFICATIONS_ENABLED`
- `DISCORD_NOTIFICATIONS_ENABLED`

Recommended defaults:

```text
MAX_TWEET_CHARS=230
MAX_RETRIES=5
LLM_TIMEOUT_SECONDS=120
NEWS_ENABLED=true
NEWS_RECENCY_HOURS=48
NEWS_REGION=US
NEWS_LANGUAGE=en
POST_TO_BLUESKY=true
BLUESKY_SERVICE_URL=https://bsky.social
POST_TO_INSTAGRAM=false
INSTAGRAM_GRAPH_BASE_URL=https://graph.instagram.com
INSTAGRAM_GRAPH_API_VERSION=v23.0
CLOUDINARY_FOLDER=content-bot
POST_TO_X=false
APPROVAL_REQUIRED=true
APPROVAL_TIMEOUT_MINUTES=90
TELEGRAM_NOTIFICATIONS_ENABLED=false
DISCORD_NOTIFICATIONS_ENABLED=false
```

For Bluesky publishing, create an app password in Bluesky for the bot account, then set:

```text
POST_TO_BLUESKY=true
BLUESKY_HANDLE=your-handle.bsky.social
BLUESKY_SERVICE_URL=https://bsky.social
BLUESKY_APP_PASSWORD=your_app_password
```

Publishing is approval-first by default. When any publishing platform is enabled and `APPROVAL_REQUIRED=true`, the bot sends a Discord approval request and waits up to `APPROVAL_TIMEOUT_MINUTES`. It publishes only after an allowed Discord user clicks Approve. Declined or expired runs are logged but not published.

Set `APPROVAL_REQUIRED=false` to skip the approval gate and publish directly to the enabled platforms. Post-publish Discord and Telegram notification formats stay the same.

For Discord approval, set:

```text
APPROVAL_REQUIRED=true
DISCORD_BOT_TOKEN=your_discord_bot_token
DISCORD_CHANNEL_ID=your_private_channel_id
DISCORD_APPROVER_USER_IDS=your_user_id,another_user_id
APPROVAL_TIMEOUT_MINUTES=90
```

For Instagram publishing, use an Instagram Creator or Business account connected to a Facebook Page, then set:

```text
POST_TO_INSTAGRAM=true
INSTAGRAM_ACCOUNT_ID=your_instagram_account_id
INSTAGRAM_GRAPH_BASE_URL=https://graph.instagram.com
INSTAGRAM_GRAPH_API_VERSION=v23.0
INSTAGRAM_ACCESS_TOKEN=your_instagram_access_token
CLOUDINARY_CLOUD_NAME=your_cloudinary_cloud_name
CLOUDINARY_API_KEY=your_cloudinary_api_key
CLOUDINARY_API_SECRET=your_cloudinary_api_secret
CLOUDINARY_FOLDER=content-bot
```

Instagram posts use a Pillow-generated square image and a caption built from the news headline, source, published time, topic/tone hashtags, LLM-suggested hashtags, and `#botWrites` as the final hashtag.

Use `INSTAGRAM_GRAPH_BASE_URL=https://graph.instagram.com` for Instagram-login tokens with `instagram_business_*` scopes. Use `https://graph.facebook.com` only if you switch to the Facebook-login Graph API flow.

`INSTAGRAM_ACCESS_TOKEN` must be the raw Meta access token only. Do not include quotes, labels, a `Bearer ` prefix, extra spaces, or copied markdown formatting. If Meta's token debugger says the token is valid but posting reports that the token cannot be parsed, verify that `INSTAGRAM_GRAPH_BASE_URL` matches the token type.

Keep `POST_TO_X=false` while X posting is not free. If Bluesky, Instagram, and X publishing are all disabled, the bot runs in manual mode and sends the final post text to notifications.

Choose `LLM_BASE_URL`, `LLM_MODEL`, and `LLM_API_KEY` based on the provider you want to use.
Do not use a provider's website URL as `LLM_BASE_URL`. For example,
`https://ollama.com` is a website, not an OpenAI-compatible API endpoint.

OpenAI-compatible provider examples:

```text
# OpenAI
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4.1-mini

# Ollama local
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=gemma3:1b

# OpenRouter
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=openai/gpt-4.1-mini

# Groq
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=llama-3.3-70b-versatile
```

## Local setup

Create a local `.env` from `.env.example`, then install dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Run once locally:

```bash
.venv/bin/python tweet_generator.py
```

GitHub cron controls the production run schedule. The Python code always runs once
when invoked.

## Logs and notifications

GitHub Actions writes successful published posts, declined approval runs, expired approval runs, and publish failures to `tweet-history.md` on a separate branch named `tweet-history`. This keeps the `main` branch stable, so routine bot runs do not create conflicts when code or workflow changes are pushed.

To view the log in GitHub, switch the branch selector from `main` to `tweet-history` and open `tweet-history.md`.

When a recent RSS item is used, the log also includes the news title, source, published time, and URL. Successful published logs include each enabled platform result. Manual-mode generated posts are not written to the history log.

For local runs, the default log path is `logs/tweet-history.md` unless `LOG_FILE_PATH` is set in `.env`.

Notification channels are opt-in. Set `TELEGRAM_NOTIFICATIONS_ENABLED=true` to send Telegram summaries, and set `DISCORD_NOTIFICATIONS_ENABLED=true` to send Discord webhook embeds. When `APPROVAL_REQUIRED=true`, approval uses `DISCORD_BOT_TOKEN` and `DISCORD_CHANNEL_ID`, not the webhook. If an enabled notification channel is missing credentials or fails to send, the bot prints a warning and continues.

Telegram requires:

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

Discord requires:

- `DISCORD_WEBHOOK_URL`

Telegram receives:

- Topic
- Tone
- Total time taken
- Number of generation attempts
- News reference, when RSS news was used
- Final post text

Discord receives the same success and failure information as a rich embed.

If a run fails before publishing a post, the workflow exits cleanly. When notification channels are enabled, the bot sends a failure summary with the topic, tone, news reference when available, and the error message. Failed runs are not written to the history log.
