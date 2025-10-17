import re
import nltk
from nltk.corpus import stopwords

class TextPreprocessor:
    def __init__(self, language='english'):
        try:
            self.stop_words = set(stopwords.words(language))
        except LookupError:
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words(language))

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        text = re.sub('<.*?>', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = text.lower()
        text = ' '.join(text.split())
        return text

    def remove_stopwords(self, text):
        tokens = text.split()
        filtered = [word for word in tokens if word not in self.stop_words]
        return ' '.join(filtered)

    def basic_features(self, text):
        words = text.split()
        n_words = len(words)
        n_chars = len(text)
        avg_word_len = sum(len(word) for word in words)/n_words if n_words else 0
        return {
            "word_count": n_words,
            "char_count": n_chars,
            "avg_word_length": avg_word_len
        }
