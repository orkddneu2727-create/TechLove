import os
import asyncio
import logging

from aiogram import Bot, Dispatcher, F, Router
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton,
    BotCommand, ReplyKeyboardMarkup, KeyboardButton,
)
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.client.default import DefaultBotProperties
from groq import AsyncGroq

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BOT_TOKEN = os.environ["BOT_TOKEN"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

groq_client = AsyncGroq(api_key=GROQ_API_KEY)

MODELS = {
    "llama4_scout": {
        "name": "Llama 4 Scout",
        "model_id": "meta-llama/llama-4-scout-17b-16e-instruct",
        "description": "Новейшая Llama 4 от Meta. Умная, быстрая, бесплатная.",
        "emoji": "🦙",
    },
    "llama3_70b": {
        "name": "Llama 3.3 70B",
        "model_id": "llama-3.3-70b-versatile",
        "description": "Мощная универсальная модель. Отлично справляется с любыми задачами.",
        "emoji": "🧠",
    },
    "llama3_8b": {
        "name": "Llama 3.1 8B",
        "model_id": "llama-3.1-8b-instant",
        "description": "Молниеносная лёгкая модель. Идеальна для быстрых ответов.",
        "emoji": "⚡",
    },
    "qwen": {
        "name": "Qwen QwQ 32B",
        "model_id": "qwen-qwq-32b",
        "description": "Мощная модель с глубоким мышлением. Отличный баланс качества и скорости.",
        "emoji": "🌀",
    },
    "deepseek_r1": {
        "name": "DeepSeek R1",
        "model_id": "deepseek-r1-distill-llama-70b",
        "description": "Модель с цепочкой рассуждений. Лучший выбор для логики и математики.",
        "emoji": "🔬",
    },
    "gemma2": {
        "name": "Gemma 2 9B",
        "model_id": "gemma2-9b-it",
        "description": "Компактная модель от Google. Быстро и качественно.",
        "emoji": "💎",
    },
}

SYSTEM_PROMPTS = {
    "default": "Ты умный, дружелюбный и полезный ИИ-ассистент. Отвечай чётко, структурировано и по делу. Используй Markdown для форматирования когда уместно.",
    "coder": "Ты опытный программист и архитектор ПО. Помогаешь писать чистый, эффективный код с подробными объяснениями. Всегда используй блоки кода с указанием языка.",
    "writer": "Ты талантливый писатель и редактор. Помогаешь с текстами, статьями, историями и копирайтингом. Пишешь живо, грамотно и увлекательно.",
    "analyst": "Ты аналитик данных и бизнес-консультант. Помогаешь анализировать информацию, строить стратегии и принимать взвешенные решения.",
    "translator": "Ты профессиональный переводчик и лингвист. Переводи точно и естественно, сохраняя стиль и смысл оригинала. После перевода можешь дать короткий комментарий если нужно.",
    "tutor": "Ты терпеливый и опытный преподаватель. Объясняешь сложные темы простым языком, используешь примеры и аналогии. Проверяешь понимание.",
}

ROLES = {
    "default":    {"name": "Ассистент",   "emoji": "🤖"},
    "coder":      {"name": "Программист", "emoji": "👨‍💻"},
    "writer":     {"name": "Писатель",    "emoji": "✍️"},
    "analyst":    {"name": "Аналитик",    "emoji": "📊"},
    "translator": {"name": "Переводчик",  "emoji": "🌐"},
    "tutor":      {"name": "Преподаватель","emoji": "🎓"},
}

user_sessions: dict[int, dict] = {}


def get_session(user_id: int) -> dict:
    if user_id not in user_sessions:
        user_sessions[user_id] = {
            "model": "llama3_70b",
            "role": "default",
            "history": [],
            "temperature": 0.7,
        }
    return user_sessions[user_id]


def main_keyboard() -> ReplyKeyboardMarkup:
    buttons = [
        [KeyboardButton(text="🤖 Модель"), KeyboardButton(text="🎭 Роль")],
        [KeyboardButton(text="⚙️ Настройки"), KeyboardButton(text="🗑 Новый диалог")],
        [KeyboardButton(text="ℹ️ Помощь")],
    ]
    return ReplyKeyboardMarkup(keyboard=buttons, resize_keyboard=True)


def models_keyboard(current: str) -> InlineKeyboardMarkup:
    buttons = []
    for key, model in MODELS.items():
        check = "✅ " if key == current else ""
        buttons.append([InlineKeyboardButton(
            text=f"{check}{model['emoji']} {model['name']}",
            callback_data=f"model:{key}"
        )])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def roles_keyboard(current: str) -> InlineKeyboardMarkup:
    buttons = []
    for key, role in ROLES.items():
        check = "✅ " if key == current else ""
        buttons.append([InlineKeyboardButton(
            text=f"{check}{role['emoji']} {role['name']}",
            callback_data=f"role:{key}"
        )])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def settings_keyboard(user_id: int) -> InlineKeyboardMarkup:
    session = get_session(user_id)
    temp = session["temperature"]
    buttons = [
        [InlineKeyboardButton(text=f"🌡 Температура: {temp:.1f}  (точно ←→ креативно)", callback_data="noop")],
        [
            InlineKeyboardButton(text="➖", callback_data="temp:down"),
            InlineKeyboardButton(text=f"{temp:.1f}", callback_data="noop"),
            InlineKeyboardButton(text="➕", callback_data="temp:up"),
        ],
        [InlineKeyboardButton(text="✅ Закрыть", callback_data="settings:close")],
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
        f"👋 Привет, *{user.first_name}*\\!\n\n"
        f"Я — ИИ\\-ассистент с доступом к лучшим *бесплатным* языковым моделям \\(Groq\\)\\.\n\n"
        f"📌 *Текущие настройки:*\n"
        f"• Модель: {model['emoji']} {model['name']}\n"
        f"• Роль: {role['emoji']} {role['name']}\n\n"
        f"Просто напиши мне сообщение — и я отвечу\\!"
    )
    await message.answer(text, parse_mode=ParseMode.MARKDOWN_V2, reply_markup=main_keyboard())


@router.message(Command("help"))
@router.message(F.text == "ℹ️ Помощь")
async def cmd_help(message: Message):
    models_text = "\n".join(
        f"• {m['emoji']} *{m['name']}* — {m['description']}" for m in MODELS.values()
    )
    roles_text = "\n".join(
        f"• {r['emoji']} *{r['name']}*" for r in ROLES.values()
    )
    text = (
        "📖 *Как пользоваться:*\n\n"
        "Просто пишите сообщение — бот отвечает с учётом истории разговора\\.\n\n"
        f"🤖 *Модели \\(все бесплатные\\):*\n{models_text}\n\n"
        f"🎭 *Роли:*\n{roles_text}\n\n"
        "⚙️ *Настройки* — регулировка температуры ответа\n"
        "🗑 *Новый диалог* — сбросить историю\n\n"
        "📝 *Команды:*\n"
        "/start — главное меню\n"
        "/new — новый диалог\n"
        "/model — сменить модель\n"
        "/role — сменить роль\n"
        "/status — текущие настройки\n"
        "/help — помощь"
    )
    await message.answer(text, parse_mode=ParseMode.MARKDOWN_V2, reply_markup=main_keyboard())


@router.message(Command("model"))
@router.message(F.text == "🤖 Модель")
async def cmd_model(message: Message):
    session = get_session(message.from_user.id)
    await message.answer(
        "🤖 *Выберите модель ИИ:*",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=models_keyboard(session["model"])
    )


@router.message(Command("role"))
@router.message(F.text == "🎭 Роль")
async def cmd_role(message: Message):
    session = get_session(message.from_user.id)
    await message.answer(
        "🎭 *Выберите роль ассистента:*",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=roles_keyboard(session["role"])
    )


@router.message(Command("new"))
@router.message(F.text == "🗑 Новый диалог")
async def cmd_new(message: Message):
    session = get_session(message.from_user.id)
    session["history"] = []
    await message.answer(
        "🗑 *История очищена\\.* Начинаем новый диалог\\!",
        parse_mode=ParseMode.MARKDOWN_V2,
        reply_markup=main_keyboard()
    )


@router.message(Command("status"))
@router.message(F.text == "⚙️ Настройки")
async def cmd_status(message: Message):
    session = get_session(message.from_user.id)
    model = MODELS[session["model"]]
    role = ROLES[session["role"]]
    text = (
        f"⚙️ *Текущие настройки:*\n\n"
        f"🤖 Модель: {model['emoji']} *{model['name']}*\n"
        f"🎭 Роль: {role['emoji']} *{role['name']}*\n"
        f"🌡 Температура: *{session['temperature']:.1f}*\n"
        f"💬 Сообщений в истории: *{len(session['history'])}*"
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
        f"✅ Модель: {model['emoji']} *{model['name']}*\n\n_{model['description']}_",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=models_keyboard(model_key)
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
        f"✅ Роль: {role['emoji']} *{role['name']}*\n\n_История очищена для применения новой роли\\._",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=roles_keyboard(role_key)
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


async def call_groq(session: dict, user_message: str) -> str:
    model_id = MODELS[session["model"]]["model_id"]
    system_prompt = SYSTEM_PROMPTS[session["role"]]

    session["history"].append({"role": "user", "content": user_message})
    if len(session["history"]) > 20:
        session["history"] = session["history"][-20:]

    messages = [{"role": "system", "content": system_prompt}] + session["history"]

    response = await groq_client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=session["temperature"],
        max_tokens=2048,
    )
    reply = response.choices[0].message.content
    session["history"].append({"role": "assistant", "content": reply})
    return reply


def escape_md(text: str) -> str:
    chars = r"_*[]()~`>#+-=|{}.!"
    for ch in chars:
        text = text.replace(ch, f"\\{ch}")
    return text


@router.message(F.text)
async def handle_message(message: Message):
    user_id = message.from_user.id
    text = message.text.strip()
    session = get_session(user_id)
    model = MODELS[session["model"]]
    role = ROLES[session["role"]]

    thinking_msg = await message.answer(
        f"⏳ _{model['emoji']} {model['name']} думает..._",
        parse_mode=ParseMode.MARKDOWN
    )

    try:
        reply = await call_groq(session, text)

        header = f"_{role['emoji']} {role['name']} · {model['name']}_\n\n"
        full_reply = header + reply

        if len(full_reply) > 4096:
            await thinking_msg.delete()
            chunks = [full_reply[i:i + 4096] for i in range(0, len(full_reply), 4096)]
            for chunk in chunks:
                await message.answer(chunk, parse_mode=ParseMode.MARKDOWN)
        else:
            await thinking_msg.edit_text(full_reply, parse_mode=ParseMode.MARKDOWN)

    except Exception as e:
        logger.error(f"Groq error for user {user_id}: {e}")
        err = str(e)[:300]
        await thinking_msg.edit_text(
            f"❌ *Ошибка:* `{err}`\n\nПопробуйте:\n• Сменить модель /model\n• Новый диалог /new",
            parse_mode=ParseMode.MARKDOWN
        )


async def set_commands(bot: Bot):
    await bot.set_my_commands([
        BotCommand(command="start",  description="Главное меню"),
        BotCommand(command="new",    description="Новый диалог"),
        BotCommand(command="model",  description="Выбрать модель ИИ"),
        BotCommand(command="role",   description="Выбрать роль ассистента"),
        BotCommand(command="status", description="Текущие настройки"),
        BotCommand(command="help",   description="Помощь"),
    ])


async def main():
    bot = Bot(
        token=BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN)
    )
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(router)
    await set_commands(bot)
    logger.info("Бот запущен на Groq!")
    await dp.start_polling(bot, allowed_updates=["message", "callback_query"])


if __name__ == "__main__":
    asyncio.run(main())
