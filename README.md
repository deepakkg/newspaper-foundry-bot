# Gemma Tweet Bot

This bot fetches recent topic news from Google News RSS, generates short posts with an OpenAI-compatible LLM API, posts them to Bluesky or X, records each successful post in GitHub, and can send Telegram and Discord notifications.

## GitHub Actions schedule

The workflow has four explicit GitHub cron entries for these India-time slots:

- 06:00 IST
- 12:00 IST
- 18:00 IST
- 22:00 IST

GitHub cron uses UTC, so the workflow file maps these to `00:30`, `06:30`, `12:30`, and `16:30` UTC.

## GitHub Secrets

Store these in repository Settings -> Secrets and variables -> Actions -> Secrets:

- `LLM_API_KEY`
- `BLUESKY_APP_PASSWORD`
- `X_API_KEY`
- `X_API_KEY_SECRET`
- `X_ACCESS_TOKEN`
- `X_ACCESS_TOKEN_SECRET`
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
- `POST_TO_X`
- `X_USERNAME`
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
POST_TO_X=false
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

When `POST_TO_BLUESKY=true`, Bluesky is the primary publisher. Keep `POST_TO_X=false` while X posting is not free. If both Bluesky and X publishing are disabled, the bot runs in manual mode and sends the final post text to notifications.

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

GitHub Actions writes successful auto-published posts to `tweet-history.md` on a separate branch named `tweet-history`. This keeps the `main` branch stable, so routine bot runs do not create conflicts when code or workflow changes are pushed.

To view the log in GitHub, switch the branch selector from `main` to `tweet-history` and open `tweet-history.md`.

When a recent RSS item is used, the log also includes the news title, source, published time, and URL. Manual-mode generated posts are not written to tweet history.

For local runs, the default log path is `logs/tweet-history.md` unless `LOG_FILE_PATH` is set in `.env`.

Notification channels are opt-in. Set `TELEGRAM_NOTIFICATIONS_ENABLED=true` to send Telegram summaries, and set `DISCORD_NOTIFICATIONS_ENABLED=true` to send Discord webhook embeds. If an enabled notification channel is missing credentials or fails to send, the bot prints a warning and continues.

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
- Full tweet text

Discord receives the same success and failure information as a rich embed.

If a run fails before posting a tweet, the workflow exits cleanly. When notification channels are enabled, the bot sends a failure summary with the topic, tone, news reference when available, and the error message. Failed runs are not written to tweet history.
