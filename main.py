"""
Telegram Expense Tracking Bot.

- Tracks expenses via SMS-style parsing and manual entry.
- Stores data in SQLite for persistence.
- Provides summaries, exports, and deletion utilities.

Environment:
- TELEGRAM_BOT_TOKEN (required)
- ADMIN_USER_ID (optional, numeric; allows /reset)
- DB_PATH (optional; default: expenses.db)
- CURRENCY_TO_MVR_RATES (optional; e.g. "USD=15.4,MYR=3.3")
- OPENAI_API_KEY & OPENAI_MODEL (optional; enable AI parsing for messy messages)
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple

import sqlite3
import tempfile
from base64 import b64encode
from telegram import (
    Update,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
    InputFile,
)
from telegram.ext import (
    ApplicationBuilder,
    Application,
    CommandHandler,
    MessageHandler,
    ConversationHandler,
    ContextTypes,
    filters,
)

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Conversation states
AMOUNT, CURRENCY, MANUAL_DATE, VENDOR, ITEMS = range(5)

DEFAULT_CURRENCY_MVR_RATES: Dict[str, float] = {
    "MVR": 1.0,
    "USD": 15.45,
    "EUR": 16.7,
    "MYR": 3.3,
    "INR": 0.2,
    "LKR": 0.05,
    "GBP": 19.6,
}

_CACHED_RATES: Optional[Dict[str, float]] = None
CURRENCY_ALIASES: Dict[str, str] = {
    "RUPEE": "INR",
    "RUPEES": "INR",
    "RS": "INR",
    "MRF": "MVR",
    "RF": "MVR",
}
AI_DEFAULT_MODEL = "gpt-4o-mini"
_AI_CLIENT: Optional["OpenAI"] = None

CATEGORY_RULES: List[Tuple[str, List[str]]] = [
    (
        "Transfer/Manual",
        [
            "bank transfer",
            "transfer",
            "manual entry",
            "cash deposit",
            "cash transfer",
            "fund transfer",
            "to account",
            "from account",
            "standing order",
        ],
    ),
    (
        "Income",
        [
            "salary",
            "deposit",
            "incoming",
            "refund",
            "rebate",
            "payroll",
            "bonus",
            "interest",
            "cashback",
            "dividend",
        ],
    ),
    (
        "Transport",
        [
            "grab",
            "uber",
            "ride",
            "taxi",
            "bus",
            "cab",
            "fuel",
            "petrol",
            "diesel",
            "parking",
            "bike",
            "ferry",
            "transport",
        ],
    ),
    (
        "Food & Dining",
        [
            "foodpanda",
            "cafe",
            "restaurant",
            "dining",
            "pizza",
            "burger",
            "kfc",
            "mcd",
            "coffee",
            "eatery",
            "bakery",
            "kitchen",
            "snack",
        ],
    ),
    (
        "Groceries",
        [
            "grocer",
            "supermarket",
            "market",
            "mart",
            "hypermarket",
            "fresh",
            "grocery",
            "wholesale",
            "carrefour",
            "lotus",
        ],
    ),
    (
        "Retail & Shopping",
        [
            "shop",
            "store",
            "mall",
            "boutique",
            "retail",
            "fashion",
            "clothing",
            "apparel",
            "brand",
            "amazon",
            "lazada",
            "shopee",
            "electronics",
            "accessory",
            "gift shop",
        ],
    ),
    (
        "Health & Wellness",
        [
            "pharmacy",
            "clinic",
            "hospital",
            "medical",
            "dental",
            "optical",
            "fitness",
            "gym",
            "wellness",
            "health",
            "lab",
        ],
    ),
    (
        "Utilities & Bills",
        [
            "utility",
            "water",
            "electric",
            "sewer",
            "rent",
            "bill",
            "internet",
            "wifi",
            "phone",
            "mobile",
            "dhiraagu",
            "ooredoo",
            "insurance",
            "premium",
            "loan",
            "mortgage",
            "gas",
            "power",
        ],
    ),
    (
        "Travel & Accommodation",
        [
            "hotel",
            "resort",
            "airbnb",
            "booking",
            "travel",
            "flight",
            "airline",
            "airasia",
            "emirates",
            "ticket",
            "visa",
        ],
    ),
    (
        "Entertainment",
        [
            "movie",
            "cinema",
            "netflix",
            "spotify",
            "show",
            "concert",
            "game",
            "gaming",
            "playstation",
            "steam",
            "event",
            "amusement",
        ],
    ),
    (
        "Education",
        [
            "school",
            "college",
            "university",
            "course",
            "tuition",
            "training",
            "exam",
            "udemy",
            "coursera",
            "bootcamp",
            "education",
            "learning",
        ],
    ),
    (
        "Tech & Subscriptions",
        [
            "saas",
            "software",
            "cloud",
            "hosting",
            "domain",
            "server",
            "github",
            "digitalocean",
            "aws",
            "gcp",
            "azure",
            "notion",
            "zoom",
            "adobe",
        ],
    ),
    (
        "Home & Services",
        [
            "repair",
            "service",
            "plumbing",
            "cleaning",
            "laundry",
            "household",
            "furniture",
            "appliance",
            "decor",
            "carpentry",
        ],
    ),
    (
        "Charity & Gifts",
        [
            "donation",
            "charity",
            "zakat",
            "gift",
            "present",
            "ngo",
            "fundraiser",
        ],
    ),
    (
        "Fees & Charges",
        [
            "fee",
            "charge",
            "penalty",
            "fine",
            "processing",
            "service charge",
        ],
    ),
]


def maybe_load_env() -> None:
    if load_dotenv:
        load_dotenv()


def iso_date_from_ddmmyy(value: str) -> str:
    """Convert DD/MM/YY to ISO YYYY-MM-DD."""
    dt = datetime.strptime(value, "%d/%m/%y")
    return dt.date().isoformat()


def iso_date_from_ddmmyyyy(value: str) -> str:
    """Convert DD/MM/YYYY to ISO YYYY-MM-DD."""
    dt = datetime.strptime(value, "%d/%m/%Y")
    return dt.date().isoformat()


def today_iso() -> str:
    return date.today().isoformat()


def categorize(text: str) -> str:
    """Rule-based categorization that tries to cover most day-to-day expenses."""
    t = (text or "").lower()
    if not t:
        return "Uncategorized"
    for category, keywords in CATEGORY_RULES:
        if any(keyword in t for keyword in keywords):
            return category
    return "Uncategorized"


def parse_rates_env(value: str) -> Dict[str, float]:
    """Parse env string like 'USD=15.4,MYR=3.29' into rates dict."""
    rates: Dict[str, float] = {}
    for part in value.split(","):
        if "=" not in part:
            continue
        code, rate_text = part.split("=", 1)
        code = code.strip().upper()
        try:
            rate = float(rate_text.strip())
        except ValueError:
            continue
        if rate > 0:
            rates[code] = rate
    return rates


def get_currency_rates() -> Dict[str, float]:
    global _CACHED_RATES
    if _CACHED_RATES is None:
        rates = dict(DEFAULT_CURRENCY_MVR_RATES)
        override = os.getenv("CURRENCY_TO_MVR_RATES")
        if override:
            rates.update(parse_rates_env(override))
        _CACHED_RATES = rates
    return _CACHED_RATES


def convert_to_mvr(amount: float, currency: Optional[str]) -> float:
    """Convert amount to MVR using static/env-configured rates."""
    code = normalize_currency(currency)
    rate = get_currency_rates().get(code)
    if not rate:
        return amount
    return amount * rate


def normalize_currency(currency: Optional[str]) -> str:
    code = (currency or "MVR").strip().upper()
    return CURRENCY_ALIASES.get(code, code)


def get_ai_client() -> Optional["OpenAI"]:
    global _AI_CLIENT
    if OpenAI is None:
        return None
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    if _AI_CLIENT is None:
        _AI_CLIENT = OpenAI(api_key=key)
    return _AI_CLIENT


def ai_parse_expense_sync(text: str) -> Optional[ParsedSMS]:
    client = get_ai_client()
    if not client:
        return None
    model = os.getenv("OPENAI_MODEL", AI_DEFAULT_MODEL)
    system_prompt = (
        "You are an assistant that extracts expense data from any finance-related text. "
        "Respond with valid JSON containing keys: amount (number), currency (3-letter code or symbol), "
        "vendor (string), date (YYYY-MM-DD or empty), time (HH:MM:SS or empty), "
        "card_last4 (string or empty). If unsure, leave a field empty. No extra commentary."
    )
    user_prompt = (
        "Extract the expense details from the following text:\n"
        f"{text}\n\n"
        "Only output JSON."
    )
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content or ""
        data = json.loads(content)
    except Exception:
        return None
    amount = data.get("amount")
    currency = data.get("currency")
    vendor = (data.get("vendor") or "Unknown").strip() or "Unknown"
    date_text = (data.get("date") or "").strip()
    time_text = (data.get("time") or "").strip()
    card_last4 = (data.get("card_last4") or "").strip()
    if amount is None or currency is None:
        return None
    try:
        amount_float = float(amount)
    except (TypeError, ValueError):
        return None
    normalized_currency = normalize_currency(str(currency))
    date_iso = today_iso()
    if date_text:
        parsed = False
        for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
            try:
                date_iso = datetime.strptime(date_text, fmt).date().isoformat()
                parsed = True
                break
            except ValueError:
                continue
        if not parsed:
            try:
                date_iso = datetime.fromisoformat(date_text).date().isoformat()
            except ValueError:
                date_iso = today_iso()
    if not time_text:
        time_text = datetime.now().strftime("%H:%M:%S")
    return ParsedSMS(
        card_last4=card_last4 or "",
        date_iso=date_iso,
        time_text=time_text,
        amount=amount_float,
        currency=normalized_currency,
        vendor=vendor,
        description=text.strip(),
    )


async def ai_parse_expense(text: str) -> Optional[ParsedSMS]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: ai_parse_expense_sync(text))


def ai_parse_image_sync(path: str) -> Optional[ParsedSMS]:
    client = get_ai_client()
    if not client:
        return None
    model = os.getenv("OPENAI_MODEL", AI_DEFAULT_MODEL)
    with open(path, "rb") as f:
        data = b64encode(f.read()).decode("utf-8")
    system_prompt = (
        "You are an assistant that reads receipts, invoices, or transfer slips. "
        "Return JSON with amount, currency, vendor, date (YYYY-MM-DD or empty), "
        "time (HH:MM:SS or empty), and card_last4 if present."
    )
    user_content = [
        {
            "type": "text",
            "text": "Extract the expense details from this image. Respond ONLY with JSON.",
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{data}"},
        },
    ]
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        message_content = response.choices[0].message.content
        if isinstance(message_content, str):
            content_text = message_content
        else:
            content_text = "".join(
                part.get("text", "")
                for part in message_content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        data = json.loads(content_text)
    except Exception:
        return None
    amount = data.get("amount")
    currency = data.get("currency")
    vendor = (data.get("vendor") or "Unknown").strip() or "Unknown"
    date_text = (data.get("date") or "").strip()
    time_text = (data.get("time") or "").strip()
    card_last4 = (data.get("card_last4") or "").strip()
    if amount is None or currency is None:
        return None
    try:
        amount_float = float(amount)
    except (TypeError, ValueError):
        return None
    normalized_currency = normalize_currency(str(currency))
    date_iso = today_iso()
    if date_text:
        parsed = False
        for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
            try:
                date_iso = datetime.strptime(date_text, fmt).date().isoformat()
                parsed = True
                break
            except ValueError:
                continue
        if not parsed:
            try:
                date_iso = datetime.fromisoformat(date_text).date().isoformat()
            except ValueError:
                date_iso = today_iso()
    if not time_text:
        time_text = datetime.now().strftime("%H:%M:%S")
    return ParsedSMS(
        card_last4=card_last4 or "",
        date_iso=date_iso,
        time_text=time_text,
        amount=amount_float,
        currency=normalized_currency,
        vendor=vendor,
        description=f"Image receipt parsed from {os.path.basename(path)}",
    )


async def ai_parse_image(path: str) -> Optional[ParsedSMS]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: ai_parse_image_sync(path))


@dataclass
class ParsedSMS:
    card_last4: str
    date_iso: str
    time_text: str
    amount: float
    currency: str
    vendor: str
    description: str


class ExpenseDB:
    def __init__(self, path: str = "expenses.db") -> None:
        self.path = path
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.init_schema()

    def init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS expenses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                amount REAL NOT NULL,
                amount_mvr REAL,
                currency TEXT,
                vendor TEXT,
                date TEXT,
                time TEXT,
                category TEXT,
                source TEXT,
                card_last4 TEXT,
                items TEXT,
                description TEXT
            )
            """
        )
        self.conn.commit()
        self.ensure_amount_mvr_column()

    def ensure_amount_mvr_column(self) -> None:
        cols = {
            row[1]
            for row in self.conn.execute("PRAGMA table_info(expenses)").fetchall()
        }
        if "amount_mvr" not in cols:
            self.conn.execute("ALTER TABLE expenses ADD COLUMN amount_mvr REAL")
            self.conn.commit()

    def add_expense(
        self,
        *,
        amount: float,
        amount_mvr: Optional[float],
        currency: str,
        vendor: str,
        date_iso: str,
        time_text: str,
        category: str,
        source: str,
        card_last4: Optional[str] = None,
        items: Optional[str] = None,
        description: Optional[str] = None,
    ) -> int:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO expenses
            (amount, amount_mvr, currency, vendor, date, time, category, source, card_last4, items, description)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                amount,
                amount_mvr if amount_mvr is not None else amount,
                currency,
                vendor,
                date_iso,
                time_text,
                category,
                source,
                card_last4,
                items,
                description,
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def delete_expense(self, expense_id: int) -> bool:
        cur = self.conn.cursor()
        cur.execute("DELETE FROM expenses WHERE id = ?", (expense_id,))
        self.conn.commit()
        return cur.rowcount > 0

    def reset(self) -> None:
        self.conn.execute("DELETE FROM expenses")
        self.conn.commit()

    def totals_by_currency(self) -> List[Tuple[str, float]]:
        rows = self.conn.execute(
            "SELECT COALESCE(currency, 'Unspecified') AS currency, SUM(amount) as total "
            "FROM expenses GROUP BY COALESCE(currency, 'Unspecified')"
        ).fetchall()
        return [(row["currency"], row["total"] or 0.0) for row in rows]

    def totals_by_category(self) -> List[Tuple[str, float]]:
        rows = self.conn.execute(
            "SELECT COALESCE(category, 'Uncategorized') AS category, SUM(COALESCE(amount_mvr, amount)) as total "
            "FROM expenses GROUP BY COALESCE(category, 'Uncategorized') ORDER BY total DESC"
        ).fetchall()
        return [(row["category"], row["total"] or 0.0) for row in rows]

    def top_vendors(self, limit: int = 5) -> List[Tuple[str, float]]:
        rows = self.conn.execute(
            "SELECT COALESCE(vendor, 'Unknown') AS vendor, SUM(COALESCE(amount_mvr, amount)) as total "
            "FROM expenses GROUP BY COALESCE(vendor, 'Unknown') "
            "ORDER BY total DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [(row["vendor"], row["total"] or 0.0) for row in rows]

    def total_all(self) -> float:
        row = self.conn.execute(
            "SELECT SUM(COALESCE(amount_mvr, amount)) as total FROM expenses"
        ).fetchone()
        return row["total"] or 0.0

    def total_month(self, year: int, month: int) -> float:
        start = date(year, month, 1).isoformat()
        end_month = month + 1
        end_year = year
        if end_month == 13:
            end_month = 1
            end_year += 1
        end = date(end_year, end_month, 1).isoformat()
        row = self.conn.execute(
            "SELECT SUM(COALESCE(amount_mvr, amount)) as total "
            "FROM expenses WHERE date >= ? AND date < ?",
            (start, end),
        ).fetchone()
        return row["total"] or 0.0

    def total_day(self, day_iso: str) -> float:
        row = self.conn.execute(
            "SELECT SUM(COALESCE(amount_mvr, amount)) as total FROM expenses WHERE date = ?",
            (day_iso,),
        ).fetchone()
        return row["total"] or 0.0

    def expenses_in_month(self, year: int, month: int) -> List[sqlite3.Row]:
        start = date(year, month, 1).isoformat()
        end_month = month + 1
        end_year = year
        if end_month == 13:
            end_month = 1
            end_year += 1
        end = date(end_year, end_month, 1).isoformat()
        return self.conn.execute(
            "SELECT * FROM expenses WHERE date >= ? AND date < ? ORDER BY date DESC, time DESC",
            (start, end),
        ).fetchall()

    def expenses_on_day(self, day_iso: str) -> List[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM expenses WHERE date = ? ORDER BY time DESC", (day_iso,)
        ).fetchall()

    def last_expenses(self, limit: int = 10) -> List[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM expenses ORDER BY date DESC, time DESC LIMIT ?", (limit,)
        ).fetchall()

    def largest_expenses(self, limit: int = 10) -> List[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM expenses ORDER BY COALESCE(amount_mvr, amount) DESC LIMIT ?",
            (limit,),
        ).fetchall()

    def all_expenses(self) -> List[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM expenses ORDER BY id ASC").fetchall()


def parse_sms_message(text: str) -> Optional[ParsedSMS]:
    pattern = re.compile(
        r"Transaction from (\d{4}) on (\d{2}/\d{2}/\d{2}) at (\d{2}:\d{2}:\d{2}) "
        r"for (MVR|MYR)(\d+(?:\.\d+)?) at (.+?) was processed",
        re.IGNORECASE,
    )
    m = pattern.search(text)
    if not m:
        return None
    card_last4 = m.group(1)
    date_text = m.group(2)
    time_text = m.group(3)
    currency = normalize_currency(m.group(4))
    amount = float(m.group(5))
    vendor = m.group(6).strip()
    try:
        date_iso = iso_date_from_ddmmyy(date_text)
    except ValueError:
        return None
    return ParsedSMS(
        card_last4=card_last4,
        date_iso=date_iso,
        time_text=time_text,
        amount=amount,
        currency=currency,
        vendor=vendor,
        description=text.strip(),
    )


def parse_bml_transfer_message(text: str) -> Optional[ParsedSMS]:
    """Parse Bank of Maldives transfer confirmations."""
    lower_text = text.lower()
    if "transfer transaction is successful" not in lower_text and "bank of maldives" not in lower_text:
        return None
    amount_match = re.search(
        r"Amount\s+(?:(?P<currency>[A-Z]{3})\s+)?(?P<amount>[\d,]+(?:\.\d+)?)",
        text,
        re.IGNORECASE,
    )
    if amount_match:
        currency = amount_match.group("currency") or "MVR"
        amount_text = amount_match.group("amount")
    else:
        fallback = re.search(
            r"([\d,]+(?:\.\d+)?)\s*(MVR|MYR|USD|EUR|GBP|INR)", text, re.IGNORECASE
        )
        if not fallback:
            return None
        amount_text = fallback.group(1)
        currency = fallback.group(2)
    try:
        amount = float(amount_text.replace(",", ""))
    except ValueError:
        return None

    dt_match = re.search(
        r"Transaction date\s+(\d{2}/\d{2}/\d{4})\s+(\d{2}:\d{2})",
        text,
        re.IGNORECASE,
    )
    if dt_match:
        raw_date = dt_match.group(1)
        raw_time = dt_match.group(2) + ":00"
        try:
            date_iso = iso_date_from_ddmmyyyy(raw_date)
        except ValueError:
            date_iso = today_iso()
        time_text = raw_time
    else:
        date_iso = today_iso()
        time_text = datetime.now().strftime("%H:%M:%S")

    remarks_match = re.search(r"Remarks\s+(.+)", text, re.IGNORECASE)
    vendor = remarks_match.group(1).strip() if remarks_match else "Bank Transfer"

    ref_match = re.search(r"Reference\s+([A-Z0-9]+)", text, re.IGNORECASE)
    card_last4 = ref_match.group(1)[-4:] if ref_match else None

    return ParsedSMS(
        card_last4=card_last4 or "",
        date_iso=date_iso,
        time_text=time_text,
        amount=amount,
        currency=normalize_currency(currency),
        vendor=vendor,
        description=text.strip(),
    )


def parse_mib_transfer_message(text: str) -> Optional[ParsedSMS]:
    """Parse Maldives Islamic Bank confirmation slips."""
    lower_text = text.lower()
    if "maldives islamic bank" not in lower_text and "reference #" not in lower_text:
        return None

    amount_match = re.search(
        r"(MVR|USD|EUR|GBP|INR)\s*([\d,]+(?:\.\d+)?)", text, re.IGNORECASE
    )
    if not amount_match:
        return None
    currency = amount_match.group(1)
    amount_text = amount_match.group(2)
    try:
        amount = float(amount_text.replace(",", ""))
    except ValueError:
        return None

    created_match = re.search(
        r"Created Date\s+(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})", text, re.IGNORECASE
    )
    if created_match:
        date_iso = created_match.group(1)
        time_text = created_match.group(2)
    else:
        status_match = re.search(
            r"Status Date\s+(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})",
            text,
            re.IGNORECASE,
        )
        if status_match:
            date_iso = status_match.group(1)
            time_text = status_match.group(2)
        else:
            date_iso = today_iso()
            time_text = datetime.now().strftime("%H:%M:%S")

    ref_match = re.search(r"Reference #\s*([A-Za-z0-9]+)", text, re.IGNORECASE)
    card_last4 = ref_match.group(1)[-4:] if ref_match else None

    purpose_match = re.search(r"Purpose\s+(.+)", text, re.IGNORECASE)
    vendor = purpose_match.group(1).strip() if purpose_match and purpose_match.group(1).strip() else "MIB Transfer"

    return ParsedSMS(
        card_last4=card_last4 or "",
        date_iso=date_iso,
        time_text=time_text,
        amount=amount,
        currency=normalize_currency(currency),
        vendor=vendor,
        description=text.strip(),
    )


def format_currency_pairs(data: List[Tuple[str, float]]) -> str:
    return ", ".join(f"{cur} {total:,.2f}" for cur, total in data) if data else "None"


def format_expense_row(row: sqlite3.Row) -> str:
    currency = (row["currency"] or "").strip()
    amount = row["amount"] or 0.0
    base_amount = row["amount_mvr"]
    parts = [
        f"#{row['id']}",
        f"{row['date']} {row['time'] or ''}".strip(),
        f"{currency} {amount:.2f}".strip(),
        row["vendor"] or "Unknown",
    ]
    if row["category"]:
        parts.append(f"[{row['category']}]")
    if base_amount is not None and (not currency or currency.upper() != "MVR" or abs(base_amount - amount) > 0.009):
        parts.append(f"(MVR {base_amount:.2f})")
    return " | ".join(parts)


def render_daily_chart(by_day: Dict[str, float]) -> List[str]:
    if not by_day:
        return ["(no data)"]
    max_total = max(by_day.values())
    scale = 20 / max_total if max_total else 1
    lines = []
    for d in sorted(by_day.keys()):
        blocks = int(by_day[d] * scale)
        bar = "#" * blocks if blocks else "."
        lines.append(f"{d}: {bar} {by_day[d]:,.2f}")
    return lines


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "Hi! I track your expenses.\n"
        "- Send bank/SMS transaction texts or receipt screenshots and I will log them.\n"
        "- Use /add for manual cash/bank entries.\n"
        "- Use /summary for totals; /month for this month; /today for today; /recent for latest entries.\n"
        "- /export to download CSV, /delete <id> to remove, /reset (admin only).\n"
        "- All amounts are stored in MVR (configure via CURRENCY_TO_MVR_RATES env).\n"
        "- /menu shows quick buttons, /closemenu hides them."
    )
    await update.message.reply_text(text)


async def menu_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = [
        ["/add", "/summary"],
        ["/month", "/today"],
        ["/recent", "/export"],
        ["/start"],
        ["/closemenu"],
    ]
    await update.message.reply_text(
        "Pick an action:",
        reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True),
    )


async def close_menu_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Menu hidden. Send /menu anytime to show it again.",
        reply_markup=ReplyKeyboardRemove(),
    )


async def add_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Enter amount:", reply_markup=ReplyKeyboardRemove())
    return AMOUNT


async def add_amount(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = (update.message.text or "").strip()
    try:
        amount = float(text)
    except ValueError:
        await update.message.reply_text("Amount must be numeric. Enter amount:")
        return AMOUNT
    context.user_data["amount"] = amount
    await update.message.reply_text("Enter currency (MVR/MYR/other):")
    return CURRENCY


async def add_currency(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    raw_currency = (update.message.text or "").strip().upper()
    currency = normalize_currency(raw_currency or "UNSPECIFIED")
    context.user_data["currency"] = currency
    await update.message.reply_text("Enter date (DD/MM/YYYY). If empty, use today:")
    return MANUAL_DATE


async def add_date(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = (update.message.text or "").strip()
    if not text:
        context.user_data["date_iso"] = today_iso()
    else:
        try:
            context.user_data["date_iso"] = iso_date_from_ddmmyyyy(text)
        except ValueError:
            await update.message.reply_text(
                "Invalid date. Use DD/MM/YYYY. Enter date (or leave blank for today):"
            )
            return MANUAL_DATE
    context.user_data["time_text"] = datetime.now().strftime("%H:%M:%S")
    await update.message.reply_text("Enter vendor (optional):")
    return VENDOR


async def add_vendor(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    vendor = (update.message.text or "").strip() or "Manual Entry"
    context.user_data["vendor"] = vendor
    await update.message.reply_text("Enter items (comma-separated) (optional):")
    return ITEMS


async def add_items(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    items = (update.message.text or "").strip()
    db: ExpenseDB = context.bot_data["db"]
    data = context.user_data
    category = categorize(f"{data.get('vendor','')} {items}")
    amount_mvr = convert_to_mvr(data["amount"], data["currency"])
    expense_id = db.add_expense(
        amount=data["amount"],
        amount_mvr=amount_mvr,
        currency=data["currency"],
        vendor=data["vendor"],
        date_iso=data["date_iso"],
        time_text=data["time_text"],
        category=category,
        source="manual",
        card_last4=None,
        items=items if items else None,
        description=None,
    )
    await update.message.reply_text(
        f"Recorded manual expense #{expense_id}:\n"
        f"{data['currency']} {data['amount']:.2f} (~MVR {amount_mvr:.2f}) "
        f"at {data['vendor']} on {data['date_iso']} ({category})."
    )
    context.user_data.clear()
    return ConversationHandler.END


async def cancel_add(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.clear()
    await update.message.reply_text("Cancelled manual entry.", reply_markup=ReplyKeyboardRemove())
    return ConversationHandler.END


async def summary_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    db: ExpenseDB = context.bot_data["db"]
    totals_cur = db.totals_by_currency()
    totals_cat = db.totals_by_category()
    vendors = db.top_vendors()
    largest = db.largest_expenses(limit=10)

    lines = [
        "========== SUMMARY ==========",
        f"All-time total (MVR): {db.total_all():,.2f}",
        "",
        "By currency (original values):",
    ]
    if totals_cur:
        lines.extend(f"  - {cur}: {total:,.2f}" for cur, total in totals_cur)
    else:
        lines.append("  - No entries yet.")
    lines.append("")
    lines.append("By category (MVR):")
    if totals_cat:
        for cat, total in totals_cat:
            lines.append(f"  - {cat}: {total:,.2f}")
    else:
        lines.append("  - No categories yet.")
    lines.append("")
    lines.append("Top vendors (MVR):")
    if vendors:
        for v, total in vendors:
            lines.append(f"  - {v}: {total:,.2f}")
    else:
        lines.append("  - No vendor data yet.")
    lines.append("")
    lines.append("Largest expenses (MVR order):")
    if largest:
        for r in largest:
            lines.append(f"  - {format_expense_row(r)}")
    else:
        lines.append("  - None recorded.")
    lines.append("")
    lines.append("Tip: use /recent to see the latest entries.")
    await update.message.reply_text("\n".join(lines))


async def month_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    db: ExpenseDB = context.bot_data["db"]
    today = date.today()
    total = db.total_month(today.year, today.month)
    rows = db.expenses_in_month(today.year, today.month)
    by_day: Dict[str, float] = {}
    grouped: Dict[str, List[sqlite3.Row]] = {}
    for row in rows:
        base_amount = row["amount_mvr"] if row["amount_mvr"] is not None else row["amount"]
        by_day[row["date"]] = by_day.get(row["date"], 0.0) + (base_amount or 0.0)
        grouped.setdefault(row["date"], []).append(row)
    lines = [
        f"{today.strftime('%B %Y')} total (MVR): {total:,.2f}",
        "Daily totals (MVR):",
    ]
    for d in sorted(by_day.keys(), reverse=True):
        lines.append(f" - {d}: {by_day[d]:,.2f}")
    lines.append("Daily chart (MVR):")
    lines.extend(f" - {line}" for line in render_daily_chart(by_day))
    if rows:
        lines.append("Entries by day:")
        for d in sorted(grouped.keys(), reverse=True):
            lines.append(f"{d}:")
            for r in grouped[d]:
                lines.append(f" - {format_expense_row(r)}")
    await update.message.reply_text("\n".join(lines))


async def today_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    db: ExpenseDB = context.bot_data["db"]
    today_value = today_iso()
    total = db.total_day(today_value)
    rows = db.expenses_on_day(today_value)
    lines = [f"Today's total (MVR): {total:,.2f}"]
    if rows:
        lines.append("Entries:")
        lines.extend(f" - {format_expense_row(r)}" for r in rows)
    await update.message.reply_text("\n".join(lines))


async def recent_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    db: ExpenseDB = context.bot_data["db"]
    rows = db.last_expenses(limit=10)
    if not rows:
        await update.message.reply_text("No expenses recorded yet.")
        return
    lines = ["======= RECENT EXPENSES ======="]
    for idx, row in enumerate(rows, start=1):
        lines.append(f"{idx:02d}. {format_expense_row(row)}")
    await update.message.reply_text("\n".join(lines))


async def handle_receipt_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not get_ai_client():
        await update.message.reply_text(
            "Image parsing requires OPENAI_API_KEY to be configured."
        )
        return
    photo = update.message.photo[-1]
    tg_file = await photo.get_file()
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        await tg_file.download_to_drive(tmp_path)
        parsed = await ai_parse_image(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    if not parsed:
        await update.message.reply_text(
            "Sorry, I could not read that image. Please try a clearer photo or enter manually with /add."
        )
        return
    db: ExpenseDB = context.bot_data["db"]
    category = categorize(parsed.vendor)
    amount_mvr = convert_to_mvr(parsed.amount, parsed.currency)
    expense_id = db.add_expense(
        amount=parsed.amount,
        amount_mvr=amount_mvr,
        currency=parsed.currency,
        vendor=parsed.vendor,
        date_iso=parsed.date_iso,
        time_text=parsed.time_text,
        category=category,
        source="image_ai",
        card_last4=parsed.card_last4 or None,
        items=None,
        description=parsed.description,
    )
    await update.message.reply_text(
        "Image parsed and recorded:\n"
        f"{parsed.currency} {parsed.amount:.2f} (~MVR {amount_mvr:.2f}) at {parsed.vendor}\n"
        f"Date: {parsed.date_iso} | Category: {category}\n"
        f"Entry id: #{expense_id}"
    )


async def export_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    db: ExpenseDB = context.bot_data["db"]
    rows = db.all_expenses()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "id",
            "amount",
            "amount_mvr",
            "currency",
            "vendor",
            "date",
            "time",
            "category",
            "source",
            "card_last4",
            "items",
            "description",
        ]
    )
    for r in rows:
        writer.writerow(
            [
                r["id"],
                r["amount"],
                r["amount_mvr"],
                r["currency"],
                r["vendor"],
                r["date"],
                r["time"],
                r["category"],
                r["source"],
                r["card_last4"],
                r["items"],
                r["description"],
            ]
        )
    output.seek(0)
    await update.message.reply_document(
        document=InputFile(output, filename="expenses.csv"),
        caption=f"Exported {len(rows)} expenses.",
    )


async def delete_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    db: ExpenseDB = context.bot_data["db"]
    parts = (update.message.text or "").split(maxsplit=1)
    if len(parts) < 2:
        await update.message.reply_text("Usage: /delete <id>")
        return
    try:
        expense_id = int(parts[1])
    except ValueError:
        await update.message.reply_text("Expense id must be a number.")
        return
    removed = db.delete_expense(expense_id)
    if removed:
        await update.message.reply_text(f"Deleted expense #{expense_id}.")
    else:
        await update.message.reply_text(f"No expense found with id {expense_id}.")


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    db: ExpenseDB = context.bot_data["db"]
    admin_id = context.bot_data.get("admin_id")
    if admin_id is None or update.effective_user.id != admin_id:
        await update.message.reply_text("Unauthorized. Only admin can reset.")
        return
    db.reset()
    await update.message.reply_text("Database cleared.")


async def handle_sms_like(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    db: ExpenseDB = context.bot_data["db"]
    text = update.message.text or ""
    parsed = parse_sms_message(text)
    ai_used = False
    if not parsed:
        parsed = parse_bml_transfer_message(text)
    if not parsed:
        parsed = parse_mib_transfer_message(text)
    if not parsed:
        parsed = await ai_parse_expense(text)
        ai_used = parsed is not None
    if not parsed:
        await update.message.reply_text(
            "I could not parse this as a transaction. Use /add to enter manually."
        )
        return
    category = categorize(parsed.vendor)
    card_last4 = parsed.card_last4 or None
    amount_mvr = convert_to_mvr(parsed.amount, parsed.currency)
    expense_id = db.add_expense(
        amount=parsed.amount,
        amount_mvr=amount_mvr,
        currency=parsed.currency,
        vendor=parsed.vendor,
        date_iso=parsed.date_iso,
        time_text=parsed.time_text,
        category=category,
        source="ai_parse" if ai_used else "sms_auto",
        card_last4=card_last4,
        items=None,
        description=parsed.description,
    )
    reply = (
        "Recorded:\n"
        f"Vendor: {parsed.vendor}\n"
        f"Amount: {parsed.currency} {parsed.amount:.2f}\n"
        f"Base (MVR): {amount_mvr:.2f}\n"
        f"Date: {parsed.date_iso}\n"
        f"Source: {'AI-assisted' if ai_used else f'Card {parsed.card_last4}'}\n"
        f"Category: {category}\n"
        f"Entry id: #{expense_id}\n\n"
        "To view totals, type /summary."
    )
    await update.message.reply_text(reply)


def build_application() -> Application:
    maybe_load_env()
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is required.")
    admin_id = os.getenv("ADMIN_USER_ID")
    admin_id_int = int(admin_id) if admin_id and admin_id.isdigit() else None
    db_path = os.getenv("DB_PATH", "expenses.db")
    db = ExpenseDB(db_path)

    app = ApplicationBuilder().token(token).build()
    app.bot_data["db"] = db
    app.bot_data["admin_id"] = admin_id_int

    conv = ConversationHandler(
        entry_points=[CommandHandler("add", add_start)],
        states={
            AMOUNT: [MessageHandler(filters.TEXT & ~filters.COMMAND, add_amount)],
            CURRENCY: [MessageHandler(filters.TEXT & ~filters.COMMAND, add_currency)],
            MANUAL_DATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, add_date)],
            VENDOR: [MessageHandler(filters.TEXT & ~filters.COMMAND, add_vendor)],
            ITEMS: [MessageHandler(filters.TEXT & ~filters.COMMAND, add_items)],
        },
        fallbacks=[CommandHandler("cancel", cancel_add)],
    )

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("menu", menu_command))
    app.add_handler(CommandHandler("closemenu", close_menu_command))
    app.add_handler(conv)
    app.add_handler(CommandHandler("summary", summary_command))
    app.add_handler(CommandHandler("month", month_command))
    app.add_handler(CommandHandler("today", today_command))
    app.add_handler(CommandHandler("recent", recent_command))
    app.add_handler(CommandHandler("export", export_command))
    app.add_handler(CommandHandler("delete", delete_command))
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(MessageHandler(filters.PHOTO, handle_receipt_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_sms_like))
    return app


def main() -> None:
    app = build_application()
    print("Bot starting polling...")
    app.run_polling()


if __name__ == "__main__":
    main()
