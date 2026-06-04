# Gemma Tweet Bot

This bot fetches recent topic news from Google News RSS, generates short tweets with Ollama, posts them to X, records each successful post in GitHub, and sends a short Telegram summary.

## GitHub Actions schedule

The workflow has four explicit GitHub cron entries for these India-time slots:

- 06:00 IST
- 12:00 IST
- 18:00 IST
- 22:00 IST

GitHub cron uses UTC, so the workflow file maps these to `00:30`, `06:30`, `12:30`, and `16:30` UTC.

## GitHub Secrets

Store these in repository Settings -> Secrets and variables -> Actions -> Secrets:

- `OLLAMA_API_KEY`
- `X_API_KEY`
- `X_API_KEY_SECRET`
- `X_ACCESS_TOKEN`
- `X_ACCESS_TOKEN_SECRET`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

## GitHub Variables

Store these in repository Settings -> Secrets and variables -> Actions -> Variables:

- `OLLAMA_HOST`
- `OLLAMA_MODEL`
- `TOPICS`
- `TONES`
- `MAX_TWEET_CHARS`
- `MAX_RETRIES`
- `OLLAMA_TIMEOUT_SECONDS`
- `NEWS_ENABLED`
- `NEWS_RECENCY_HOURS`
- `NEWS_REGION`
- `NEWS_LANGUAGE`
- `POST_TO_X`
- `X_USERNAME`
- `RUN_TIMEZONE`

Recommended defaults:

```text
OLLAMA_HOST=https://ollama.com
OLLAMA_MODEL=gemma4:31b-cloud
MAX_TWEET_CHARS=230
MAX_RETRIES=5
OLLAMA_TIMEOUT_SECONDS=120
NEWS_ENABLED=true
NEWS_RECENCY_HOURS=48
NEWS_REGION=US
NEWS_LANGUAGE=en
POST_TO_X=true
RUN_TIMEZONE=Asia/Kolkata
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

Run with the GitHub schedule guard:

```bash
.venv/bin/python tweet_generator.py --respect-schedule
```

## Logs and Telegram

GitHub Actions writes successful posts to `tweet-history.md` on a separate branch named `tweet-history`. This keeps the `main` branch stable, so routine bot runs do not create conflicts when code or workflow changes are pushed.

To view the log in GitHub, switch the branch selector from `main` to `tweet-history` and open `tweet-history.md`.

When a recent RSS item is used, the log also includes the news title, source, published time, and URL.

For local runs, the default log path is `logs/tweet-history.md` unless `LOG_FILE_PATH` is set in `.env`.

Telegram receives only:

- Topic
- Tone
- Total time taken
- Number of generation attempts
- News reference, when RSS news was used
- Full tweet text

If a run fails before posting a tweet, the workflow exits cleanly. When Telegram is configured, the bot sends a failure summary with the topic, tone, news reference when available, and the error message. Failed runs are not written to tweet history.
