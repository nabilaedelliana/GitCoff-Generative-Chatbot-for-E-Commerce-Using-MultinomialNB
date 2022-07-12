import numpy as np
import string
import logging
import pyfiglet
import logging.config
import os
import re
import joblib
from util import JSONParser
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters                  

# Function for Processing Chat from user
def chat_processing(chat):
    # Transform Chat Into Lowercase
    chat = chat.lower()

    # Remove Punctuation From Chat
    chat = chat.translate(str.maketrans("","",string.punctuation))

    # Remove Digit From Chat
    chat = re.sub("[^A-Za-z\s']"," ", chat)

    # Remove Tab From Chat
    chat = chat.strip()

    # Stemmer Definition
    stemmer = StemmerFactory().create_stemmer()

    # Stemming Chat
    chat = stemmer.stem(chat)

    return chat

def response(chat, pipeline, jp):
    chat = chat_processing(chat)
    res = pipeline.predict_proba([chat])
    max_prob = max(res[0])
    if max_prob < 0.2:
        return "Mohon maaf nih kak, aku masih belum ngerti maksud kakak :(" , None
    else:
        max_id = np.argmax(res[0])
        pred_tag = pipeline.classes_[max_id]
        return jp.get_response(pred_tag), pred_tag

def start(update, context):
    update.message.reply_text("Selamat!, kakak telah terhubung dengan Gitcoff, sebuah Chatbot AI dari Git Coffee 😉")

def respons(update, context):
    chat = update.message.text

    # Load dataset Intents for Bot Responses
    path = "../dataset/intents.json"
    jp = JSONParser()
    jp.parse(path)
    df = jp.get_dataframe()

    # Load Chatbot Machine Learning Model
    model = joblib.load("chatbot.pkl")

    res, tag = response(chat, model, jp)
    update.message.reply_text(res)

def error(update, context):
    """Log Errors caused by Updates."""
    logging.warning('Update "%s" ', update)
    logging.exception(context.error)

def main():
    updater = Updater(DefaultConfig.TELEGRAM_TOKEN, use_context=True)

    dp = updater.dispatcher

    # Command handlers
    dp.add_handler(CommandHandler("start",start))

    # Message handler
    dp.add_handler(MessageHandler(Filters.text, respons))

    # log all errors
    dp.add_error_handler(error)
    
    # Start the Bot
    if DefaultConfig.MODE == 'webhook':

        updater.start_webhook(listen="0.0.0.0",
                              port=int(DefaultConfig.PORT),
                              url_path=DefaultConfig.TELEGRAM_TOKEN)
        updater.bot.setWebhook(DefaultConfig.WEBHOOK_URL + DefaultConfig.TELEGRAM_TOKEN)
        

        logging.info(f"Start webhook mode on port {DefaultConfig.PORT}")
    else:
        updater.start_polling()
        logging.info(f"Start polling mode")

    updater.idle()

class DefaultConfig:
    PORT = int(os.environ.get("PORT", 3978))
    TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "YourBotTokenAPI")
    MODE = os.environ.get("MODE", "polling")
    WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "YourHerokuLink")

    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()

    @staticmethod
    def init_logging():
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                            level=DefaultConfig.LOG_LEVEL)
        #logging.config.fileConfig('logging.conf')

if __name__ == '__main__':
    ascii_banner = pyfiglet.figlet_format("AdvancedTelegramBot")
    print(ascii_banner)

    # Enable logging
    DefaultConfig.init_logging()

    main()