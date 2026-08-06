import os
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NLTK_DATA = os.path.join(BASE_DIR, "nltk_data")
nltk.data.path.insert(0, NLTK_DATA)

stop_words = set(stopwords.words("english"))

def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\w*\d\w*", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = wordpunct_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]

    return " ".join(tokens)