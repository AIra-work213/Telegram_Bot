from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
import TG_bot.token as t

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("")

app = Application.builder().token(t.TOKEN).build()
app.add_handler(CommandHandler("start", start))

app.run_polling()