# Expense Tracker Telegram Bot

Telegram bot that logs expenses from:
- Bank/SMS transaction texts (BML, MIB, regex-based messages)
- Messy free-form messages via OpenAI parsing
- Receipt screenshots/photos (OpenAI vision)
- Manual `/add` workflow

It persists entries in SQLite, converts every amount to MVR, and exposes summaries, exports, and clean-up commands.

## Requirements

- Python 3.10+
- Telegram bot token from [@BotFather](https://t.me/BotFather)
- (Optional) OpenAI API key for AI parsing of unformatted text/photos

## Quick Start

```bash
git clone <repo-url>
cd Expense\ Tracker\ BOT
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env           # then fill it in
python main.py
```

The bot will start polling; keep the process running. Stop with `Ctrl+C`.

## Environment Variables

Set these in `.env` (the app calls `load_dotenv()`):

| Key | Required | Description |
| --- | --- | --- |
| `TELEGRAM_BOT_TOKEN` | ✅ | Bot token from BotFather |
| `ADMIN_USER_ID` | optional | Telegram user ID allowed to run `/reset` |
| `DB_PATH` | optional | Path to SQLite DB (default `expenses.db`) |
| `CURRENCY_TO_MVR_RATES` | optional | Override conversion rates, e.g. `USD=15.4,MYR=3.3` |
| `OPENAI_API_KEY` | optional | Enables AI parsing for free-form text/photos |
| `OPENAI_MODEL` | optional | OpenAI model name (default `gpt-4o-mini`) |

## Commands

- `/start` – help text and features
- `/add` – multi-step manual entry (amount → currency → date → vendor → items)
- `/summary` – formatted totals, currencies, categories, top vendors, largest expenses
- `/month` – current month totals with simple chart
- `/today` – today's totals and entries
- `/recent` – last 10 expenses
- `/export` – CSV download
- `/delete <id>` – remove an entry
- `/reset` – admin-only purge
- `/menu` & `/closemenu` – show/hide quick keyboard buttons

The bot also reacts to:
- Plain text transaction messages
- Receipt/transfers photos (requires OpenAI key)

## Docker

```bash
docker build -t expense-bot .
docker run -it --rm \
  -e TELEGRAM_BOT_TOKEN=123:ABC \
  -e OPENAI_API_KEY=sk-... \
  -v $PWD/data:/app/data \
  -e DB_PATH=/app/data/expenses.db \
  expense-bot
```

## Testing the AI Features

1. Ensure `OPENAI_API_KEY` (and optional `OPENAI_MODEL`) are set.
2. Restart `python main.py`.
3. Forward a messy transaction string; the reply should include `Source: AI-assisted`.
4. Send a receipt photo; the bot should parse it and respond with recorded details.

## Notes

- `.gitignore` excludes `.env`, `.venv`, SQLite databases, and uploads.
- Default currency conversions and category rules are defined in `main.py`; adjust as needed.
