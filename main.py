import os
import asyncio
import logging
from typing import Optional

from aiogram import Bot, Dispatcher, F, Router
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton,
    BotCommand, ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove
)
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.client.default import DefaultBotProperties
from openai import AsyncOpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BOT_TOKEN = os.environ["BOT_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

MODELS = {
    "gpt4o": {
        "name": "GPT-4o",
        "model_id": "gpt-4o",
        "description": "Самая умная модель OpenAI. Отлично справляется со сложными задачами, анализом и кодом.",
        "emoji": "🧠",
    },
    "gpt4o_mini": {
        "name": "GPT-4o mini",
        "model_id": "gpt-4o-mini",
        "description": "Быстрая и дешёвая модель. Идеальна для повседневных задач.",
        "emoji": "⚡",
    },
    "o1_mini": {
        "name": "o1-mini",
        "model_id": "o1-mini",
        "description": "Модель с расширенным мышлением. Лучший выбор для математики и логики.",
        "emoji": "🔬",
    },
}

SYSTEM_PROMPTS = {
    "default": "Ты умный, дружелюбный и полезный ИИ-ассистент. Отвечай чётко, структурировано и по делу. Используй Markdown для форматирования когда это уместно.",
    "coder": "Ты опытный программист и архитектор ПО. Помогаешь писать чистый, эффективный код с подробными объяснениями. Всегда используй блоки кода с указанием языка.",
    "writer": "Ты талантливый писатель и редактор. Помогаешь с текстами, статьями, историями и копирайтингом. Пишешь живо, грамотно и увлекательно.",
    "analyst": "Ты аналитик данных и бизнес-консультант. Помогаешь анализировать информацию, строить стратегии и принимать взвешенные решения.",
    "translator": "Ты профессиональный переводчик и лингвист. Переводишь тексты точно и естественно, сохраняя стиль и смысл оригинала.",
}

ROLES = {
    "default": {"name": "Ассистент", "emoji": "🤖"},
    "coder": {"name": "Программист", "emoji": "👨‍💻"},
    "writer": {"name": "Писатель", "emoji": "✍️"},
    "analyst": {"name": "Аналитик", "emoji": "📊"},
    "translator": {"name": "Переводчик", "emoji": "🌐"},
}

user_sessions: dict[int, dict] = {}


def get_session(user_id: int) -> dict:
    if user_id not in user_sessions:
        user_sessions[user_id] = {
            "model": "gpt4o_mini",
            "role": "default",
            "history": [],
            "temperature": 0.7,
        }
    return user_sessions[user_id]


def main_keyboard() -> ReplyKeyboardMarkup:
    buttons = [
        [KeyboardButton(text="🤖 Выбрать модель"), KeyboardButton(text="🎭 Выбрать роль")],
        [KeyboardButton(text="⚙️ Настройки"), KeyboardButton(text="🗑 Очистить историю")],
        [KeyboardButton(text="ℹ️ Помощь")],
    ]
    return ReplyKeyboardMarkup(keyboard=buttons, resize_keyboard=True)


def models_keyboard() -> InlineKeyboardMarkup:
    buttons = []
    for key, model in MODELS.items():
        buttons.append([
            InlineKeyboardButton(
                text=f"{model['emoji']} {model['name']}",
                callback_data=f"model:{key}"
            )
        ])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def roles_keyboard() -> InlineKeyboardMarkup:
    buttons = []
    for key, role in ROLES.items():
        buttons.append([
            InlineKeyboardButton(
                text=f"{role['emoji']} {role['name']}",
                callback_data=f"role:{key}"
            )
        ])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def settings_keyboard(user_id: int) -> InlineKeyboardMarkup:
    session = get_session(user_id)
    temp = session["temperature"]
    buttons = [
        [
            InlineKeyboardButton(text="🌡 Температура:", callback_data="noop"),
        ],
        [
            InlineKeyboardButton(text="➖", callback_data="temp:down"),
            InlineKeyboardButton(text=f"{temp:.1f}", callback_data="noop"),
            InlineKeyboardButton(text="➕", callback_data="temp:up"),
        ],
        [InlineKeyboardButton(text="✅ Готово", callback_data="settings:close")],
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)


router = Router()


@router.message(CommandStart())
async def cmd_start(message: Message):
    user = message.from_user
    session = get_session(user.id)
    model = MODELS[session["model"]]
    role = ROLES[session["role"]]

    text = (
        f"👋 Привет, *{user.first_name}*!\n\n"
        f"Я — мощный ИИ-ассистент с доступом к лучшим языковым моделям.\n\n"
        f"📌 *Текущие настройки:*\n"
        f"• Модель: {model['emoji']} {model['name']}\n"
        f"• Роль: {role['emoji']} {role['name']}\n\n"
        f"Просто напиши мне сообщение — и я отвечу. Используй меню ниже для смены модели или режима."
    )
    await message.answer(text, parse_mode=ParseMode.MARKDOWN, reply_markup=main_keyboard())


@router.message(Command("help"))
@router.message(F.text == "ℹ️ Помощь")
async def cmd_help(message: Message):
    text = (
        "📖 *Как пользоваться ботом:*\n\n"
        "• Просто напиши сообщение — бот ответит с помощью выбранной ИИ-модели\n"
        "• Бот помнит контекст разговора (историю сообщений)\n\n"
        "🤖 *Доступные модели:*\n"
        + "\n".join(f"• {m['emoji']} *{m['name']}* — {m['description']}" for m in MODELS.values())
        + "\n\n"
        "🎭 *Доступные роли:*\n"
        + "\n".join(f"• {r['emoji']} *{r['name']}*" for r in ROLES.values())
        + "\n\n"
        "⚙️ *Настройки:* регулировка температуры (0.0 — точно, 1.0 — креативно)\n\n"
        "🗑 *Очистить историю* — начать разговор заново\n\n"
        "📝 *Команды:*\n"
        "/start — главное меню\n"
        "/new — новый диалог\n"
        "/model — сменить модель\n"
        "/role — сменить роль\n"
        "/status — текущие настройки\n"
        "/help — помощь"
    )
    await message.answer(text, parse_mode=ParseMode.MARKDOWN, reply_markup=main_keyboard())


@router.message(Command("model"))
@router.message(F.text == "🤖 Выбрать модель")
async def cmd_model(message: Message):
    await message.answer(
        "🤖 *Выберите модель ИИ:*\n\nКаждая модель имеет свои сильные стороны.",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=models_keyboard()
    )


@router.message(Command("role"))
@router.message(F.text == "🎭 Выбрать роль")
async def cmd_role(message: Message):
    await message.answer(
        "🎭 *Выберите роль ассистента:*\n\nРоль определяет стиль и специализацию ответов.",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=roles_keyboard()
    )


@router.message(Command("new"))
@router.message(F.text == "🗑 Очистить историю")
async def cmd_new(message: Message):
    session = get_session(message.from_user.id)
    session["history"] = []
    await message.answer(
        "🗑 *История очищена.* Начинаем новый диалог!\n\nПишите — я готов помочь.",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=main_keyboard()
    )


@router.message(Command("status"))
@router.message(F.text == "⚙️ Настройки")
async def cmd_status(message: Message):
    session = get_session(message.from_user.id)
    model = MODELS[session["model"]]
    role = ROLES[session["role"]]
    history_count = len(session["history"])

    text = (
        f"⚙️ *Текущие настройки:*\n\n"
        f"🤖 Модель: {model['emoji']} *{model['name']}*\n"
        f"🎭 Роль: {role['emoji']} *{role['name']}*\n"
        f"🌡 Температура: *{session['temperature']:.1f}*\n"
        f"💬 Сообщений в истории: *{history_count}*\n\n"
        f"_{model['description']}_"
    )
    await message.answer(
        text,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=settings_keyboard(message.from_user.id)
    )


@router.callback_query(F.data.startswith("model:"))
async def cb_model(callback: CallbackQuery):
    model_key = callback.data.split(":")[1]
    session = get_session(callback.from_user.id)
    session["model"] = model_key
    model = MODELS[model_key]
    await callback.message.edit_text(
        f"✅ Модель изменена на {model['emoji']} *{model['name']}*\n\n_{model['description']}_",
        parse_mode=ParseMode.MARKDOWN
    )
    await callback.answer(f"Выбрана: {model['name']}")


@router.callback_query(F.data.startswith("role:"))
async def cb_role(callback: CallbackQuery):
    role_key = callback.data.split(":")[1]
    session = get_session(callback.from_user.id)
    session["role"] = role_key
    session["history"] = []
    role = ROLES[role_key]
    await callback.message.edit_text(
        f"✅ Роль изменена на {role['emoji']} *{role['name']}*\n\n"
        f"История очищена для применения новой роли.",
        parse_mode=ParseMode.MARKDOWN
    )
    await callback.answer(f"Роль: {role['name']}")


@router.callback_query(F.data.startswith("temp:"))
async def cb_temp(callback: CallbackQuery):
    action = callback.data.split(":")[1]
    session = get_session(callback.from_user.id)
    if action == "up":
        session["temperature"] = min(1.0, round(session["temperature"] + 0.1, 1))
    elif action == "down":
        session["temperature"] = max(0.0, round(session["temperature"] - 0.1, 1))
    await callback.message.edit_reply_markup(reply_markup=settings_keyboard(callback.from_user.id))
    await callback.answer(f"Температура: {session['temperature']:.1f}")


@router.callback_query(F.data == "settings:close")
async def cb_settings_close(callback: CallbackQuery):
    await callback.message.delete()
    await callback.answer()


@router.callback_query(F.data == "noop")
async def cb_noop(callback: CallbackQuery):
    await callback.answer()


async def call_openai(session: dict, user_message: str) -> str:
    model_key = session["model"]
    model_id = MODELS[model_key]["model_id"]
    role_key = session["role"]
    system_prompt = SYSTEM_PROMPTS[role_key]

    session["history"].append({"role": "user", "content": user_message})

    if len(session["history"]) > 20:
        session["history"] = session["history"][-20:]

    messages = [{"role": "system", "content": system_prompt}] + session["history"]

    kwargs = {
        "model": model_id,
        "messages": messages,
    }

    if model_key not in ("o1_mini",):
        kwargs["temperature"] = session["temperature"]
        kwargs["max_tokens"] = 2000

    response = await openai_client.chat.completions.create(**kwargs)
    reply = response.choices[0].message.content

    session["history"].append({"role": "assistant", "content": reply})

    return reply


@router.message(F.text)
async def handle_message(message: Message):
    user_id = message.from_user.id
    text = message.text.strip()

    session = get_session(user_id)
    model = MODELS[session["model"]]
    role = ROLES[session["role"]]

    thinking_msg = await message.answer(
        f"⏳ *{model['emoji']} {model['name']}* думает...",
        parse_mode=ParseMode.MARKDOWN
    )

    try:
        reply = await call_openai(session, text)

        header = f"_{role['emoji']} {role['name']} · {model['name']}_\n\n"
        full_reply = header + reply

        if len(full_reply) > 4096:
            chunks = [full_reply[i:i+4096] for i in range(0, len(full_reply), 4096)]
            await thinking_msg.delete()
            for chunk in chunks:
                await message.answer(chunk, parse_mode=ParseMode.MARKDOWN)
        else:
            await thinking_msg.edit_text(full_reply, parse_mode=ParseMode.MARKDOWN)

    except Exception as e:
        logger.error(f"OpenAI error for user {user_id}: {e}")
        error_text = (
            "❌ *Ошибка при обращении к ИИ.*\n\n"
            f"Детали: `{str(e)[:200]}`\n\n"
            "Попробуйте:\n"
            "• Сменить модель (/model)\n"
            "• Очистить историю (/new)\n"
            "• Попробовать позже"
        )
        await thinking_msg.edit_text(error_text, parse_mode=ParseMode.MARKDOWN)


async def set_commands(bot: Bot):
    commands = [
        BotCommand(command="start", description="Главное меню"),
        BotCommand(command="new", description="Новый диалог"),
        BotCommand(command="model", description="Выбрать модель ИИ"),
        BotCommand(command="role", description="Выбрать роль ассистента"),
        BotCommand(command="status", description="Текущие настройки"),
        BotCommand(command="help", description="Помощь"),
    ]
    await bot.set_my_commands(commands)


async def main():
    bot = Bot(
        token=BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN)
    )
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(router)

    await set_commands(bot)
    logger.info("Бот запущен!")

    await dp.start_polling(bot, allowed_updates=["message", "callback_query"])


if __name__ == "__main__":
    asyncio.run(main())
