"""Telegram-бот адресных алертов AralGuard (aiogram v3).

Принцип анти-alert-fatigue: пользователь подписан на СВОЙ район,
получает только оранжевый/красный уровни + отбой. Никакого спама.

Запуск: TELEGRAM_BOT_TOKEN в .env, затем python -m aralguard.bot.bot
"""
import asyncio
import json
import os
from pathlib import Path

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import (InlineKeyboardButton, InlineKeyboardMarkup, Message,
                           CallbackQuery)
from dotenv import load_dotenv

from aralguard.config import DISTRICTS

SUBS_FILE = Path(__file__).resolve().parents[3] / "data_store" / "subs.json"
dp = Dispatcher()

LEVEL_EMOJI = {"orange": "🟠", "red": "🔴", "clear": "🟢"}


def load_subs() -> dict:
    return json.loads(SUBS_FILE.read_text()) if SUBS_FILE.exists() else {}


def save_subs(s: dict):
    SUBS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SUBS_FILE.write_text(json.dumps(s, ensure_ascii=False, indent=1))


def district_kb() -> InlineKeyboardMarkup:
    rows, row = [], []
    for name in DISTRICTS:
        row.append(InlineKeyboardButton(text=name, callback_data=f"d:{name}"))
        if len(row) == 3:
            rows.append(row); row = []
    if row:
        rows.append(row)
    return InlineKeyboardMarkup(inline_keyboard=rows)


@dp.message(Command("start"))
async def start(m: Message):
    await m.answer(
        "AralGuard — chang bo'roni va sel haqida erta ogohlantirish.\n"
        "Tumaningizni tanlang — faqat SIZGA tegishli ogohlantirish keladi:",
        reply_markup=district_kb())


@dp.callback_query(F.data.startswith("d:"))
async def subscribe(cb: CallbackQuery):
    district = cb.data[2:]
    subs = load_subs()
    subs[str(cb.from_user.id)] = district
    save_subs(subs)
    await cb.message.edit_text(
        f"✅ {district} tumaniga obuna bo'ldingiz.\n"
        "Faqat 🟠/🔴 daraja va yakuniy «xavf o'tdi» xabari keladi.")
    await cb.answer()


async def broadcast_alert(bot: Bot, district: str, level: str, eta_h: int,
                          pm: int, action: str):
    """Вызывается пайплайном прогноза при пересечении порога района."""
    text = (f"{LEVEL_EMOJI[level]} {district.upper()} — "
            f"{'CHANG BO`RONI' if level != 'clear' else 'XAVF O`TDI'}\n"
            f"Yetib kelishi: ~{eta_h} soat · kutilayotgan PM10: {pm} µg/m³\n"
            f"Tavsiya: {action}")
    for uid, d in load_subs().items():
        if d == district:
            await bot.send_message(int(uid), text)


async def main():
    load_dotenv()
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("TELEGRAM_BOT_TOKEN не задан в .env")
    bot = Bot(token)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
