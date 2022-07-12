# import library
import string
import pickle
import numpy as np
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from util import JSONParser

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

# Load dataset Intents
path = "dataset/intents.json"
jp = JSONParser()
jp.parse(path)
df = jp.get_dataframe()

# Preprocessing Chat Text with Case Folding
df['chat_input_prep'] = df.chat_input.apply(chat_processing)

# Modeling
nb_pipeline = make_pipeline(CountVectorizer(),
                            MultinomialNB())

# Train Model
print("[Info] GitCoff is studying our language...")
nb_pipeline.fit(df.chat_input_prep, df.intents)

# Save Model for deployment
with open("chatbot.pkl", "wb") as model_file:
    pickle.dump(nb_pipeline, model_file)

# interaction with bot
print("[Info] Selamat!, kakak telah terhubung dengan Gitcoff, sebuah Chatbot AI dari Git Coffee")
while True:
    chat = input("Anda    >> ")
    res, tag = response(chat, nb_pipeline, jp)
    print(f"Gitcoff >> {res}")
    if tag == 'menutup':
        break
