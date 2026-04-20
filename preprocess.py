import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data silently
nltk.download('stopwords', quiet=True)
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)

# Load English stopwords once at module level (efficient)
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Full preprocessing pipeline for a raw news article string.
    Returns a cleaned, space-joined string of meaningful tokens.
    """

    # Guard: if input is not a string (e.g. NaN), return empty
    if not isinstance(text, str):
        return ""

    # 1. Lowercase everything
    text = text.lower()

    # 2. Remove content inside square brackets e.g. [VIDEO], [PHOTO]
    text = re.sub(r'\[.*?\]', '', text)

    # 3. Remove URLs (http, https, www)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # 4. Remove HTML tags
    text = re.sub(r'<.*?>+', '', text)

    # 5. Remove punctuation
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)

    # 6. Remove newlines and replace with space
    text = re.sub(r'\n', ' ', text)

    # 7. Remove words containing digits (e.g. "2020", "19th", "mp4")
    text = re.sub(r'\w*\d\w*', '', text)

    # 8. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # 9. Tokenize into individual words
    tokens = word_tokenize(text)

    # 10. Remove stopwords and very short tokens (length <= 2)
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]

    # 11. Rejoin into a clean string
    return ' '.join(tokens)


# Test when this file is run directly
if __name__ == "__main__":
    test_cases = [
        "BREAKING: Donald Trump said on Tuesday that https://news.com elections were RIGGED in 2024!!!",
        "Scientists have discovered a new vaccine that cures all diseases. <p>Click here!</p>",
        "The president of the United States held a press conference about the economy.",
        None,   # test edge case
        "",     # test empty string
    ]

    for t in test_cases:
        print(f"Original : {t}")
        print(f"Cleaned  : {clean_text(t)}")
        print()